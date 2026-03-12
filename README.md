# Master Thesis: VLA + 3D Semantic Scene Understanding

Augmenting Vision-Language-Action models with UNITE 3D features for object navigation in HM3D (Habitat).

## Slides

https://LSailer.github.io/Master-Thesis-3D-VLA/

## Ralph (Autonomous TDD Agent)

```bash
# Start Ralph (picks up to N ready AFK issues, implements via TDD, creates PRs)
gh workflow run ralph.yml -f max_tasks=5 -f time_limit=120

# Check run status
gh run list --workflow=ralph.yml

# See which issues Ralph is working on / finished
gh issue list --label in-progress
gh issue list --label in-review

# See ready tasks waiting for Ralph
gh issue list --label AFK --label ready

# Promote backlog issues to ready (manual)
gh issue edit <N> --remove-label backlog --add-label ready
```
