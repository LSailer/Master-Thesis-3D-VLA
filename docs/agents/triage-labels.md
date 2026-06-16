# Triage Labels

The skills speak in terms of five canonical triage roles. This repo maps agent delegation roles to Linear labels plus workflow state.

Known Linear labels for `3D-WM-ObjectNAV` include `experiment-run`, `Feature`, `Bug`, and `Improvement`; these are type/context labels, not triage roles.

| Role in mattpocock/skills | Linear mapping | Meaning |
| ------------------------- | -------------- | ------- |
| `needs-triage` | State `Backlog` | Maintainer needs to evaluate the issue. |
| `needs-info` | Blocker relation or comment; leave workflow state unchanged unless another rule applies | Waiting on reporter or external information. |
| `ready-for-agent` | Linear label `ready-for-agent` and state `Backlog` or `Todo` | Fully specified, ready for an AFK agent. |
| `needs-human` | Linear label `needs-human`; use blocker relations when it blocks another issue | Requires human implementation, judgement, credentials, manual validation, or an unblocking decision. |
| `wontfix` | State `Canceled` or equivalent closed/canceled state | Will not be actioned. |

When applying a role, update the Linear state, apply the relevant delegation label when one exists, and add a concise comment that explains the transition. Use `needs-human` for human escalation; do not use `ready-for-human`.

If an agent cannot continue an issue because of an actionable human blocker, create or link the blocking `needs-human` subissue and mark the current issue as blocked by that subissue.

When all blocking `needs-human` subissues are completed, the automation should requeue the blocked parent automatically: remove or ignore the resolved blocker relations, move the parent to `Todo`, keep or re-add `ready-for-agent`, and add a comment explaining that the human blockers were resolved.
