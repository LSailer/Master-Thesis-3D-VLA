# Claude Code Setup Guide

Replicate this Claude Code configuration on any machine.

## Prerequisites

- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) installed
- Node.js (for MCP servers via `npx`)
- Python + `uv` (for Python-based tools)
- `jq` (for status line script)

## 1. User Settings (`~/.claude/settings.json`)

This is the **global** config — applies to all projects on this machine.

```json
{
  "permissions": {
    "allow": [
      "Bash(*)",
      "Read",
      "Write",
      "Edit",
      "Glob",
      "Grep",
      "WebSearch",
      "WebFetch"
    ],
    "deny": [
      "Read(**/.env*)",
      "Read(**/secrets/**)",
      "Read(**/*.pem)",
      "Read(**/*.key)",
      "Read(~/.ssh/**)",
      "Read(~/.aws/**)"
    ],
    "ask": [
      "Bash(rm -rf *)",
      "Bash(git push --force *)",
      "Bash(git reset --hard *)",
      "Bash(git checkout -- *)",
      "Bash(git clean *)",
      "Bash(git branch -D *)",
      "Bash(sudo rm *)",
      "Bash(chmod 777 *)"
    ]
  },
  "enableAllProjectMcpServers": true,
  "enabledPlugins": {
    "superpowers@claude-plugins-official": true,
    "context7@claude-plugins-official": true,
    "code-simplifier@claude-plugins-official": true,
    "feature-dev@claude-plugins-official": true,
    "playwright@claude-plugins-official": true,
    "pr-review-toolkit@claude-plugins-official": true,
    "pyright-lsp@claude-plugins-official": true,
    "claude-code-setup@claude-plugins-official": true,
    "explanatory-output-style@claude-plugins-official": true,
    "huggingface-skills@claude-plugins-official": true
  },
  "effortLevel": "high",
  "statusLine": {
    "type": "command",
    "command": "bash ~/.claude/statusline-command.sh"
  }
}
```

### What each section does

| Section | Purpose |
|---------|---------|
| `allow` | `Bash(*)` auto-approves all shell commands; built-in tools listed individually |
| `deny` | Blocks reading secrets (`.env`, SSH keys, PEM files) |
| `ask` | Forces confirmation on destructive commands (`rm -rf`, `git push --force`, etc.) |
| `enabledPlugins` | Pre-installed plugin set (see Plugins section below) |
| `effortLevel` | `"high"` = deeper thinking on complex tasks |
| `statusLine` | Custom PS1-style status bar showing user@host, model, context usage |

## 2. Status Line Script (`~/.claude/statusline-command.sh`)

```bash
#!/usr/bin/env bash
input=$(cat)
cwd=$(echo "$input" | jq -r '.workspace.current_dir // .cwd')
host=$(hostname -s)
user=$(whoami)
dir=$(basename "$cwd")
model=$(echo "$input" | jq -r '.model.display_name // empty')
used=$(echo "$input" | jq -r '.context_window.used_percentage // empty')

status="[${user}@${host} ${dir}]"
[ -n "$model" ] && status="${status} ${model}"
[ -n "$used" ] && status="${status} ctx:$(printf '%.0f' "$used")%"

printf "%s" "$status"
```

Make it executable:
```bash
chmod +x ~/.claude/statusline-command.sh
```

Output example: `[ul_hfj15@laptop Master-Thesis-3D-VLA] Claude Opus 4.6 ctx:23%`

## 3. Project Settings (`.claude/settings.json`)

Already checked into the repo. Minimal — defers to user settings:

```json
{
  "permissions": {
    "allow": [
      "Bash(*)",
      "Read",
      "Write",
      "Edit",
      "Glob",
      "Grep",
      "WebSearch",
      "WebFetch"
    ]
  }
}
```

## 4. MCP Servers (`.mcp.json`)

Already checked into the repo at project root:

```json
{
  "mcpServers": {
    "sequentialthinking": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sequentialthinking"]
    }
  }
}
```

To add more (optional):
```json
{
  "mcpServers": {
    "sequentialthinking": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sequentialthinking"]
    },
    "arxiv": {
      "command": "uvx",
      "args": ["arxiv-mcp-server"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_PERSONAL_ACCESS_TOKEN": "<your-token>" }
    }
  }
}
```

## 5. Plugins

Plugins are declared in user settings (`enabledPlugins`). On first launch, Claude Code downloads them automatically.

| Plugin | Purpose |
|--------|---------|
| `superpowers` | TDD, systematic debugging, plan execution, brainstorming, code review workflows |
| `context7` | Live documentation lookup for libraries |
| `code-simplifier` | Auto-simplify code for clarity |
| `feature-dev` | Architecture analysis and guided feature development |
| `playwright` | Browser automation / testing |
| `pr-review-toolkit` | PR review with specialized agents |
| `pyright-lsp` | Python type checking and navigation |
| `claude-code-setup` | Automation recommendations |
| `explanatory-output-style` | Educational insights in responses |
| `huggingface-skills` | HuggingFace training, tracking, datasets |

## 6. Custom Skills (`.claude/skills/`)

Already in the repo under `.claude/skills/`. Available skills:

| Skill | Invoke with | Purpose |
|-------|-------------|---------|
| `grill-me` | `/grill-me` | Stress-test a plan or design with relentless questioning |
| `tdd` | `/tdd` | Test-driven development workflow |
| `slurm-submit` | `/slurm-submit` | Submit SLURM GPU jobs on BWUniCluster |
| `write-a-prd` | `/write-a-prd` | Write a product requirements document |
| `prd-to-issues` | `/prd-to-issues` | Convert PRD into GitHub issues |
| `qa-review` | `/qa-review` | Quality assurance review |
| `ralph-loop` | `/ralph-loop` | Autonomous task execution loop |

## 7. Environment Variables (optional)

Add to user settings under `"env"` if desired:

```json
{
  "env": {
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "64000",
    "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": "80",
    "MCP_TIMEOUT": "30000",
    "MCP_TOOL_TIMEOUT": "60000"
  }
}
```

| Variable | Value | Purpose |
|----------|-------|---------|
| `CLAUDE_CODE_MAX_OUTPUT_TOKENS` | `64000` | Double output budget (default 32K) — prevents truncation |
| `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` | `80` | Compact old messages at 80% context usage |
| `MCP_TIMEOUT` | `30000` | 30s MCP server startup timeout |
| `MCP_TOOL_TIMEOUT` | `60000` | 60s per MCP tool call |

## Quick Setup Script

Run this on a new machine after cloning the repo:

```bash
# 1. Create ~/.claude directory
mkdir -p ~/.claude

# 2. Copy user settings (edit additionalDirectories for your paths)
cat > ~/.claude/settings.json << 'EOF'
{
  "permissions": {
    "allow": [
      "Bash(*)",
      "Read",
      "Write",
      "Edit",
      "Glob",
      "Grep",
      "WebSearch",
      "WebFetch"
    ],
    "deny": [
      "Read(**/.env*)",
      "Read(**/secrets/**)",
      "Read(**/*.pem)",
      "Read(**/*.key)",
      "Read(~/.ssh/**)",
      "Read(~/.aws/**)"
    ],
    "ask": [
      "Bash(rm -rf *)",
      "Bash(git push --force *)",
      "Bash(git reset --hard *)",
      "Bash(git checkout -- *)",
      "Bash(git clean *)",
      "Bash(git branch -D *)",
      "Bash(sudo rm *)",
      "Bash(chmod 777 *)"
    ]
  },
  "enableAllProjectMcpServers": true,
  "enabledPlugins": {
    "superpowers@claude-plugins-official": true,
    "context7@claude-plugins-official": true,
    "code-simplifier@claude-plugins-official": true,
    "feature-dev@claude-plugins-official": true,
    "playwright@claude-plugins-official": true,
    "pr-review-toolkit@claude-plugins-official": true,
    "pyright-lsp@claude-plugins-official": true,
    "claude-code-setup@claude-plugins-official": true,
    "explanatory-output-style@claude-plugins-official": true,
    "huggingface-skills@claude-plugins-official": true
  },
  "effortLevel": "high",
  "statusLine": {
    "type": "command",
    "command": "bash ~/.claude/statusline-command.sh"
  }
}
EOF

# 3. Copy status line script
cat > ~/.claude/statusline-command.sh << 'SCRIPT'
#!/usr/bin/env bash
input=$(cat)
cwd=$(echo "$input" | jq -r '.workspace.current_dir // .cwd')
host=$(hostname -s)
user=$(whoami)
dir=$(basename "$cwd")
model=$(echo "$input" | jq -r '.model.display_name // empty')
used=$(echo "$input" | jq -r '.context_window.used_percentage // empty')

status="[${user}@${host} ${dir}]"
[ -n "$model" ] && status="${status} ${model}"
[ -n "$used" ] && status="${status} ctx:$(printf '%.0f' "$used")%"

printf "%s" "$status"
SCRIPT
chmod +x ~/.claude/statusline-command.sh

# 4. Done — project settings, skills, and MCP are already in the repo
echo "Setup complete. Run 'claude' in the project directory."
```

## Notes

- **Project-level configs** (`.claude/settings.json`, `.claude/skills/`, `.mcp.json`, `CLAUDE.md`) travel with the repo — no action needed after `git clone`.
- **User-level configs** (`~/.claude/settings.json`, `~/.claude/statusline-command.sh`) must be set up per machine — use the quick setup script above.
- **`additionalDirectories`** in user settings are machine-specific paths — remove or update them for your laptop.
- Plugins download automatically on first `claude` launch.
