# Triage Labels

The skills speak in terms of five canonical triage roles. This repo maps those roles to Linear workflow state and comments, not dedicated labels.

Known Linear labels for `3D-WM-ObjectNAV` include `experiment-run`, `Feature`, `Bug`, and `Improvement`; these are type/context labels, not triage roles.

| Role in mattpocock/skills | Linear mapping | Meaning |
| ------------------------- | -------------- | ------- |
| `needs-triage` | State `Backlog` | Maintainer needs to evaluate the issue. |
| `needs-info` | Blocked state if available; otherwise comment and leave in current state | Waiting on reporter or external information. |
| `ready-for-agent` | State `Backlog` or `Todo` with an agent-ready brief/comment | Fully specified, ready for an AFK agent. |
| `ready-for-human` | State `Backlog` or `Todo` with a human-required note | Requires human implementation or judgement. |
| `wontfix` | State `Canceled` or equivalent closed/canceled state | Will not be actioned. |

When applying a role, prefer updating the Linear state and adding a concise comment that explains the transition. Do not create new triage labels unless the user asks for literal label-based triage.
