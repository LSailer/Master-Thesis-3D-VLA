# Triage Labels

The skills speak in terms of five canonical triage roles. This file maps those roles to the actual label strings used in this repo's issue tracker.

This repo uses the **pure canonical vocabulary** — each role's label string equals its role name.

| Label in mattpocock/skills | Label in our tracker | Meaning                                  |
| -------------------------- | -------------------- | ---------------------------------------- |
| `needs-triage`             | `needs-triage`       | Maintainer needs to evaluate this issue  |
| `needs-info`               | `needs-info`         | Waiting on reporter for more information |
| `ready-for-agent`          | `ready-for-agent`    | Fully specified, ready for an AFK agent  |
| `ready-for-human`          | `ready-for-human`    | Requires human implementation            |
| `wontfix`                  | `wontfix`            | Will not be actioned                     |

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), use the corresponding label string from this table.

## Note on adjacent labels

This repo also uses `AFK` and `HITL` labels, but they describe **execution mode** (autonomous vs human-in-the-loop run), not triage state. They are orthogonal to the canonical triage labels above and managed by the auto-research pipeline, not the `triage` skill. Don't conflate them.
