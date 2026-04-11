---
name: optimize
description: After a full pipeline run (/engineer → /qa → /reporter), review which skills underperformed and use OpenSpace to improve them. Use at the end of a pipeline cycle.
---

Review the last pipeline run and optimize the weakest skill using OpenSpace.

## When invoked

1. Ask the user: which step had issues or felt slow? (engineer, qa, reporter, review — or "auto" to analyze all)
2. If "auto", review the conversation for:
   - How many QA feedback loops happened (more loops = engineer skill needs improvement)
   - Whether review found many blockers (engineer skill needs stricter conventions)
   - Whether reporter output needed manual corrections (reporter skill needs refinement)

## Optimization Steps

### 1. Search for better patterns

```
search_skills(query="<description of what underperformed>", source="all")
```

Check if the community has a proven skill for the weak area. If found, read its SKILL.md and incorporate the best parts into your local skill.

### 2. Fix the skill

```
fix_skill(
  skill_dir=".claude/skills/<skill-name>",
  direction="<specific improvement based on what went wrong>"
)
```

Be specific in the direction. Examples:
- "Engineer skill missed JAX shape validation — add a checklist step to verify tensor shapes before committing"
- "QA skill didn't catch missing type hints — add type hint checking to validation checklist"
- "Reporter skill produced plots without proper axis labels — enforce thesis formatting standards"

### 3. Verify the improvement

Read the updated SKILL.md and confirm the change makes sense.

### 4. Upload if generally useful

```
upload_skill(
  skill_dir=".claude/skills/<skill-name>",
  visibility="public"
)
```

Only upload if the improvement would help others. Project-specific fixes stay local.

## Print summary

- Which skill was optimized
- What was changed and why
- Whether it was uploaded to the community
- Recommendation for the next pipeline run
