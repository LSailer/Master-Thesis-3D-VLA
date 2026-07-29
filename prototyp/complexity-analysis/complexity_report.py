"""Combined function-level and architecture-level complexity report for `src`.

Function complexity comes from radon (cyclomatic / McCabe, same numbers as
`radon cc`). Architecture complexity is derived from a static import graph
built with `ast`, giving per-module afferent/efferent coupling, instability,
import cycles and - the metric that matters most here - the blast radius, i.e.
how many modules transitively depend on a module.

The two axes are crossed into a risk score, so that a module is only called a
hotspot when it is both internally complex and widely depended upon.

Run it with radon injected, so the shared .venv is never re-synced:

    uvx --with radon python prototyp/complexity-analysis/complexity_report.py
"""

from __future__ import annotations

import ast
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from radon.complexity import cc_visit
from radon.visitors import Function

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "src"
PACKAGE_NAME = "src"
REPORT_PATH = REPO_ROOT / "outputs" / "prototype" / "complexity-analysis" / "report.md"

# A module counts as "high" on an axis when it sits in the top third of that
# axis. Crossing the two thirds is what separates a hotspot from contained
# complexity.
HIGH_PERCENTILE = 2 / 3


@dataclass
class Module:
    """Static facts about one Python module inside the package.

    Attributes:
        name: Dotted module name, e.g. `src.buffer.replay_buffer`.
        path: Absolute path of the source file.
        imports: Dotted names of package-internal modules it imports.
        functions: Radon function records for every function and method in it.
        importers: Dotted names of package-internal modules importing it.
    """

    name: str
    path: Path
    imports: set[str] = field(default_factory=set)
    functions: list[Function] = field(default_factory=list)
    importers: set[str] = field(default_factory=set)

    @property
    def decisions(self) -> int:
        """Total decision points in the module (sum of complexity - 1)."""
        return sum(fn.complexity - 1 for fn in self.functions)

    @property
    def max_complexity(self) -> int:
        """Cyclomatic complexity of the most complex single function."""
        return max((fn.complexity for fn in self.functions), default=0)

    @property
    def worst_function(self) -> Function | None:
        """The most complex function in the module, or None if it has none."""
        return max(self.functions, key=lambda fn: fn.complexity, default=None)

    @property
    def efferent(self) -> int:
        """Fan-out: number of internal modules this module depends on."""
        return len(self.imports)

    @property
    def afferent(self) -> int:
        """Fan-in: number of internal modules depending on this module."""
        return len(self.importers)

    @property
    def instability(self) -> float:
        """Martin instability Ce / (Ca + Ce); 0.0 when the module is isolated."""
        total = self.afferent + self.efferent
        return self.efferent / total if total else 0.0


def module_name_for(path: Path) -> str:
    """Map a source path to its dotted module name.

    Args:
        path: Path to a `.py` file inside the package root.

    Returns:
        The dotted module name, with `__init__.py` collapsing to its package.
    """
    relative = path.relative_to(PACKAGE_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join([PACKAGE_NAME, *parts])


def discover_modules() -> dict[str, Module]:
    """Collect every module in the package, skipping caches and vendored trees.

    Returns:
        Mapping of dotted module name to a Module with path set.
    """
    modules: dict[str, Module] = {}
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        name = module_name_for(path)
        modules[name] = Module(name=name, path=path)
    return modules


def resolve_target(raw: str, known: set[str]) -> str | None:
    """Resolve an imported dotted name to the nearest known module.

    `from src.buffer import replay_buffer` names a submodule, while
    `from src.buffer import SomeClass` names an attribute of the package. Both
    arrive here as candidate dotted names, so the longest known prefix wins.

    Args:
        raw: Dotted name as written in the import statement.
        known: All module names in the package.

    Returns:
        The matching module name, or None if the import leaves the package.
    """
    parts = raw.split(".")
    while parts:
        candidate = ".".join(parts)
        if candidate in known:
            return candidate
        parts.pop()
    return None


def collect_imports(module: Module, tree: ast.AST, known: set[str]) -> None:
    """Record every package-internal import of a module on the module itself.

    Args:
        module: The module being analysed; mutated in place.
        tree: Parsed AST of that module.
        known: All module names in the package.
    """
    package = module.name.rsplit(".", 1)[0] if "." in module.name else module.name
    for node in ast.walk(tree):
        candidates: list[str] = []
        if isinstance(node, ast.Import):
            candidates = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_parts = package.split(".")
                trimmed = base_parts[: len(base_parts) - node.level + 1]
                base = ".".join(trimmed)
            else:
                base = node.module or ""
            if not base:
                continue
            candidates = [base, *(f"{base}.{alias.name}" for alias in node.names)]

        for raw in candidates:
            if not raw.startswith(PACKAGE_NAME):
                continue
            target = resolve_target(raw, known)
            if target and target != module.name:
                module.imports.add(target)


def analyse(modules: dict[str, Module]) -> None:
    """Fill in imports and function complexity for every module.

    Args:
        modules: Discovered modules, mutated in place.
    """
    known = set(modules)
    for module in modules.values():
        source = module.path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(module.path))
        except SyntaxError as exc:
            print(f"skipping {module.name}: {exc}", file=sys.stderr)
            continue
        collect_imports(module, tree, known)
        module.functions = [
            block for block in cc_visit(source) if isinstance(block, Function)
        ]

    for module in modules.values():
        for target in module.imports:
            modules[target].importers.add(module.name)


def blast_radius(modules: dict[str, Module]) -> dict[str, int]:
    """Count how many modules transitively depend on each module.

    Args:
        modules: Fully analysed modules.

    Returns:
        Mapping of module name to the size of its transitive dependent set.
    """
    radius: dict[str, int] = {}
    for name in modules:
        seen: set[str] = set()
        queue = [name]
        while queue:
            current = queue.pop()
            for importer in modules[current].importers:
                if importer not in seen:
                    seen.add(importer)
                    queue.append(importer)
        radius[name] = len(seen)
    return radius


def find_cycles(modules: dict[str, Module]) -> list[list[str]]:
    """Find import cycles as strongly connected components of size > 1.

    Args:
        modules: Fully analysed modules.

    Returns:
        Sorted list of cycles, each a sorted list of module names.
    """
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    counter = 0
    components: list[list[str]] = []

    def strong_connect(name: str) -> None:
        nonlocal counter
        index[name] = low[name] = counter
        counter += 1
        stack.append(name)
        on_stack.add(name)
        for target in sorted(modules[name].imports):
            if target not in index:
                strong_connect(target)
                low[name] = min(low[name], low[target])
            elif target in on_stack:
                low[name] = min(low[name], index[target])
        if low[name] == index[name]:
            component = []
            while True:
                node = stack.pop()
                on_stack.discard(node)
                component.append(node)
                if node == name:
                    break
            if len(component) > 1:
                components.append(sorted(component))

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10 * len(modules) + 1000))
    for name in sorted(modules):
        if name not in index:
            strong_connect(name)
    return sorted(components, key=len, reverse=True)


def threshold(values: list[float]) -> float:
    """Return the cut-off separating the top third of a distribution.

    Args:
        values: Observed values on one axis.

    Returns:
        The value at the HIGH_PERCENTILE position, or 0.0 for an empty input.
    """
    positives = sorted(v for v in values if v > 0)
    if not positives:
        return 0.0
    position = min(int(len(positives) * HIGH_PERCENTILE), len(positives) - 1)
    return positives[position]


def classify(decisions: int, radius: int, cut_dec: float, cut_rad: float) -> str:
    """Assign a module to a quadrant of the complexity/reach plane.

    Args:
        decisions: Total decision points in the module.
        radius: Number of modules transitively depending on it.
        cut_dec: Complexity cut-off for "high".
        cut_rad: Blast-radius cut-off for "high".

    Returns:
        One of `hotspot`, `contained`, `hub`, `ok`.
    """
    complex_enough = decisions >= cut_dec and decisions > 0
    central_enough = radius >= cut_rad and radius > 0
    if complex_enough and central_enough:
        return "hotspot"
    if complex_enough:
        return "contained"
    if central_enough:
        return "hub"
    return "ok"


def render(modules: dict[str, Module]) -> str:
    """Render the full markdown report.

    Args:
        modules: Fully analysed modules.

    Returns:
        The report as a markdown string.
    """
    radius = blast_radius(modules)
    cycles = find_cycles(modules)
    cut_dec = threshold([float(m.decisions) for m in modules.values()])
    cut_rad = threshold([float(r) for r in radius.values()])

    rows = []
    for module in modules.values():
        risk = module.decisions * (1 + radius[module.name])
        rows.append((risk, module, radius[module.name]))
    rows.sort(key=lambda row: row[0], reverse=True)
    peak = rows[0][0] if rows and rows[0][0] else 1

    total_functions = sum(len(m.functions) for m in modules.values())
    total_edges = sum(m.efferent for m in modules.values())

    lines = [
        "# Complexity report: function level x architecture level",
        "",
        f"- Modules: {len(modules)}",
        f"- Functions and methods: {total_functions}",
        f"- Internal import edges: {total_edges}",
        f"- Import cycles: {len(cycles)}",
        "",
        "Risk = decision points in the module x (1 + blast radius), normalised",
        "to 100. A module scores high only when complex code sits behind a wide",
        "set of dependents. Quadrant thresholds are the top third of each axis:",
        f"decisions >= {cut_dec:.0f}, blast radius >= {cut_rad:.0f}.",
        "",
        "## Top 20 by risk",
        "",
        "| Risk | Module | Decisions | Max CC | Worst function | Ca | Ce | I | Blast | Quadrant |",
        "|---:|---|---:|---:|---|---:|---:|---:|---:|---|",
    ]

    for risk, module, reach in rows[:20]:
        worst = module.worst_function
        worst_label = (
            f"`{worst.name}` ({worst.complexity})" if worst and worst.complexity > 1 else "-"
        )
        quadrant = classify(module.decisions, reach, cut_dec, cut_rad)
        lines.append(
            f"| {100 * risk / peak:.0f} | `{module.name}` | {module.decisions} | "
            f"{module.max_complexity} | {worst_label} | {module.afferent} | "
            f"{module.efferent} | {module.instability:.2f} | {reach} | {quadrant} |"
        )

    lines += ["", "## Quadrants", ""]
    buckets: dict[str, list[str]] = defaultdict(list)
    for _, module, reach in rows:
        buckets[classify(module.decisions, reach, cut_dec, cut_rad)].append(module.name)
    labels = {
        "hotspot": "complex and widely depended upon - refactor here first",
        "contained": "complex but few dependents - safe to rework in isolation",
        "hub": "simple but widely depended upon - keep it that way",
        "ok": "neither complex nor central",
    }
    for key in ("hotspot", "contained", "hub", "ok"):
        names = buckets[key]
        lines.append(f"- **{key}** ({len(names)}): {labels[key]}")
        if key != "ok":
            for name in names:
                lines.append(f"  - `{name}`")

    lines += ["", "## Import cycles", ""]
    if cycles:
        for cycle in cycles:
            lines.append(f"- {' -> '.join(f'`{name}`' for name in cycle)}")
    else:
        lines.append("None. The import graph is a DAG.")

    lines += ["", "## Dependency graph of the top 10 risk modules", "", "```mermaid", "graph LR"]
    top_names = [module.name for _, module, _ in rows[:10]]
    aliases = {name: f"n{i}" for i, name in enumerate(top_names)}
    for name in top_names:
        short = name.removeprefix(f"{PACKAGE_NAME}.")
        lines.append(f"  {aliases[name]}[\"{short}\"]")
    for name in top_names:
        for target in sorted(modules[name].imports):
            if target in aliases:
                lines.append(f"  {aliases[name]} --> {aliases[target]}")
    lines += ["```", ""]

    return "\n".join(lines)


def main() -> None:
    """Build the report, write it to disk and print a short summary."""
    modules = discover_modules()
    analyse(modules)
    report = render(modules)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nwritten to {REPORT_PATH.relative_to(REPO_ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
