#!/usr/bin/env python3
"""SessionEnd hook: analyze transcript for skill failures and improvements.

Reads the session transcript (JSONL), identifies skills that were invoked,
detects runtime errors or manual fixes that followed skill execution,
and appends lessons learned to the relevant SKILL.md files.

Changes are committed automatically.
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_hook_input():
    """Read JSON from stdin (SessionEnd hook payload)."""
    try:
        data = json.loads(sys.stdin.read())
        return data.get("transcript_path"), data.get("cwd", os.getcwd())
    except (json.JSONDecodeError, EOFError):
        return None, os.getcwd()


def parse_transcript(transcript_path: str) -> list[dict]:
    """Parse JSONL transcript into a list of messages."""
    messages = []
    try:
        with open(transcript_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    messages.append(json.loads(line))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return messages


def find_skill_invocations(messages: list[dict]) -> dict[str, list[str]]:
    """Find which skills were invoked and collect errors that followed.

    Returns: {skill_name: [error_messages]}
    """
    skills_used = {}
    current_skill = None

    for msg in messages:
        text = json.dumps(msg) if isinstance(msg, dict) else str(msg)

        # Detect skill invocation
        skill_match = re.search(r'skill["\s:]+["\']?(\w[\w-]*)', text, re.IGNORECASE)
        if skill_match:
            current_skill = skill_match.group(1).lower()
            if current_skill not in skills_used:
                skills_used[current_skill] = []

        # Detect errors after a skill was invoked
        if current_skill:
            error_patterns = [
                r'(?:Error|Exception|Traceback|FAIL|error).*?:.*?(.{20,200})',
                r'exit code [1-9]\d*',
            ]
            for pattern in error_patterns:
                error_match = re.search(pattern, text)
                if error_match:
                    error_text = error_match.group(0)[:200]
                    if error_text not in skills_used[current_skill]:
                        skills_used[current_skill].append(error_text)

    return skills_used


def find_skill_file(skill_name: str, skills_dir: Path) -> Path | None:
    """Find the SKILL.md file for a given skill name."""
    skill_path = skills_dir / skill_name / "SKILL.md"
    if skill_path.exists():
        return skill_path
    # Try fuzzy match
    for d in skills_dir.iterdir():
        if d.is_dir() and skill_name in d.name.lower():
            candidate = d / "SKILL.md"
            if candidate.exists():
                return candidate
    return None


def append_lesson(skill_path: Path, errors: list[str], date: str):
    """Append a lessons-learned section to a SKILL.md file."""
    content = skill_path.read_text()

    # Don't duplicate — check if these errors are already recorded
    errors = [e for e in errors if e[:80] not in content]

    if not errors:
        return False

    lesson_block = f"\n\n## Lessons learned ({date})\n\n"
    lesson_block += "Issues encountered during session:\n"
    for error in errors[:5]:  # Cap at 5 lessons per session
        # Clean up the error for readability
        clean = error.strip().replace("\n", " ")[:150]
        lesson_block += f"- `{clean}`\n"

    skill_path.write_text(content + lesson_block)
    return True


def git_commit_skills(cwd: str, modified: list[str]):
    """Commit modified skill files."""
    if not modified:
        return

    try:
        for f in modified:
            subprocess.run(
                ["git", "add", f],
                cwd=cwd, capture_output=True, timeout=10
            )
        subprocess.run(
            ["git", "commit", "-m",
             f"chore: auto-improve skills from session ({datetime.now():%Y-%m-%d})"],
            cwd=cwd, capture_output=True, timeout=10
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass


def main():
    transcript_path, cwd = parse_hook_input()
    if not transcript_path or not os.path.exists(transcript_path):
        return

    messages = parse_transcript(transcript_path)
    if not messages:
        return

    skills_used = find_skill_invocations(messages)
    if not skills_used:
        return

    skills_dir = Path(cwd) / ".claude" / "skills"
    if not skills_dir.exists():
        return

    date = datetime.now().strftime("%Y-%m-%d")
    modified = []

    for skill_name, errors in skills_used.items():
        if not errors:
            continue

        skill_path = find_skill_file(skill_name, skills_dir)
        if skill_path and append_lesson(skill_path, errors, date):
            modified.append(str(skill_path))

    git_commit_skills(cwd, modified)


if __name__ == "__main__":
    main()
