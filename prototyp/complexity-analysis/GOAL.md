# Goal

Cross function-level complexity with architecture-level complexity, so that
"this code is complex" and "a lot depends on this code" can be read as one
number instead of two separate tool outputs.

## Hypothesis

Neither axis alone ranks refactoring targets correctly. A function with high
cyclomatic complexity in a leaf module is cheap to rework, while a moderately
complex module that half the package imports is expensive to touch. Only the
product identifies where complexity actually hurts.

## Approach

`complexity_report.py` does both passes over `src` in one AST walk per file:

- **Function level**: radon `cc_visit`, the same cyclomatic numbers as
  `radon cc` and `ruff --select C901`. Per module it aggregates decision
  points (sum of complexity - 1), the maximum, and the worst function.
- **Architecture level**: a static import graph over package-internal imports,
  giving afferent coupling (Ca), efferent coupling (Ce), Martin instability
  I = Ce / (Ca + Ce), Tarjan strongly connected components for import cycles,
  and the blast radius, i.e. the transitive dependent count.

Risk = decision points x (1 + blast radius), normalised to 100. Quadrants come
from the top third of each axis:

| Quadrant | Meaning |
|---|---|
| hotspot | complex and widely depended upon - refactor here first |
| contained | complex but few dependents - safe to rework in isolation |
| hub | simple but widely depended upon - keep it that way |
| ok | neither |

## Class level

`class_metrics.py` adds the third granularity, between function and module,
using the CK metric suite:

- **CBO** (coupling between objects): distinct other project classes a class
  references. Outward coupling, the "how hard is this to move or test" number.
- **LCOM4** (lack of cohesion of methods): connected components of the method
  graph, where two methods are linked when they share an instance attribute or
  one calls the other. Inward decoupling: LCOM4 = n means n unrelated
  responsibilities share a namespace, and each component is its own candidate
  class.
- Supporting: WMC (sum of method complexities), DIT, NOC.

LCOM4 only means something once the constructs that are isolated *by
construction* are removed, otherwise the metric is dominated by false
positives. Excluded are `__init__`/`__post_init__`, staticmethods and
classmethods, methods reading no instance state, and declaration-only classes
(Protocol, ABC, Enum, TypedDict, NamedTuple, stub-only bodies).

## Run

```bash
uvx --with radon python prototyp/complexity-analysis/complexity_report.py
```

```bash
uvx --with radon python prototyp/complexity-analysis/class_metrics.py
```

radon comes through `uvx --with`, following the repo convention that tools
which only read the code never re-sync the shared `.venv`. The reports are
written to `outputs/prototype/complexity-analysis/`.
