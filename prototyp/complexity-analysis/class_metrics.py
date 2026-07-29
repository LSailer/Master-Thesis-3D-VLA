"""Class-level coupling and cohesion metrics for `src` (CK metric suite).

Two directions are measured, because "decoupling" at class level means both:

- **CBO** (coupling between objects): how many other project classes a class
  references. High CBO means the class is hard to move, test or reuse.
- **LCOM4** (lack of cohesion of methods): the number of connected components
  in the method graph, where two methods are connected when they share an
  instance attribute or one calls the other. LCOM4 = 1 is a cohesive class,
  LCOM4 = n means the class is n unrelated classes sharing a namespace.

Supporting metrics are WMC (sum of method complexities, from radon), DIT
(depth of inheritance) and NOC (number of children).

`__init__` and `__post_init__` are excluded from the LCOM4 graph. A constructor
touches every attribute by definition and would collapse every class to a
single component, hiding exactly what the metric is supposed to expose.
Staticmethods are excluded too, since they never touch `self` and would show up
as isolated components regardless of how well the class hangs together.

Run it with radon injected, so the shared .venv is never re-synced:

    uvx --with radon python prototyp/complexity-analysis/class_metrics.py
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

from radon.complexity import cc_visit
from radon.visitors import Class as RadonClass

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "src"
PACKAGE_NAME = "src"
REPORT_PATH = (
    REPO_ROOT / "outputs" / "prototype" / "complexity-analysis" / "class_report.md"
)

EXCLUDED_FROM_COHESION = {"__init__", "__post_init__"}

# Bases that make a class a declaration rather than an implementation. Their
# methods share no state by design, so LCOM4 would flag every single one.
INTERFACE_BASES = {
    "Protocol",
    "ABC",
    "ABCMeta",
    "Enum",
    "IntEnum",
    "StrEnum",
    "Flag",
    "IntFlag",
    "TypedDict",
    "NamedTuple",
}


@dataclass
class ClassInfo:
    """Measured facts about one class definition.

    Attributes:
        name: Bare class name.
        module: Dotted module name it is defined in.
        lineno: Line of the `class` statement.
        bases: Bare names of its base classes.
        methods: Names of the methods that entered the cohesion graph.
        stateless: Methods that read no instance state, excluded from LCOM4.
        wmc: Weighted methods per class, the sum of method complexities.
        max_cc: Cyclomatic complexity of its most complex method.
        components: Connected components of the method graph, largest first.
        cbo: Number of distinct other project classes it references.
        noc: Number of direct subclasses inside the project.
        interface: True when the class only declares an API and implements
            nothing, so its cohesion is not meaningful.
    """

    name: str
    module: str
    lineno: int
    bases: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    stateless: list[str] = field(default_factory=list)
    wmc: int = 0
    max_cc: int = 0
    components: list[list[str]] = field(default_factory=list)
    cbo: int = 0
    noc: int = 0
    interface: bool = False

    @property
    def qualified(self) -> str:
        """Module-qualified class name."""
        return f"{self.module}.{self.name}"

    @property
    def lcom4(self) -> int:
        """Lack of cohesion: the number of connected method components."""
        return len(self.components)

    @property
    def islands(self) -> list[list[str]]:
        """Components beyond the largest one, i.e. the detachable method sets."""
        return self.components[1:]


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


def source_files() -> list[Path]:
    """List every analysable source file in the package.

    Returns:
        Sorted paths, excluding bytecode caches.
    """
    return sorted(
        path
        for path in PACKAGE_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def base_name(node: ast.expr) -> str | None:
    """Reduce a base-class expression to its bare name.

    Args:
        node: Expression node from a `ClassDef.bases` entry.

    Returns:
        The bare name, e.g. `Module` for `nn.Module`, or None if not a name.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def self_references(method: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Collect every `self.<name>` touched inside a method.

    Args:
        method: The method definition to inspect.

    Returns:
        Names accessed on `self`, whether attributes or method calls.
    """
    names: set[str] = set()
    for node in ast.walk(method):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            names.add(node.attr)
    return names


def is_unbound(method: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Report whether a method never binds the instance.

    Both `@staticmethod` and `@classmethod` receive no `self`, so they cannot
    participate in instance-attribute cohesion and would otherwise register as
    isolated components no matter how well the class is designed.

    Args:
        method: The method definition to inspect.

    Returns:
        True when the method is a staticmethod or a classmethod.
    """
    return any(
        isinstance(dec, ast.Name) and dec.id in ("staticmethod", "classmethod")
        for dec in method.decorator_list
    )


def is_stub(method: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Report whether a method body is a placeholder rather than an implementation.

    Args:
        method: The method definition to inspect.

    Returns:
        True when the body is only a docstring, `pass` and/or `...`.
    """
    for statement in method.body:
        if isinstance(statement, ast.Pass):
            continue
        if isinstance(statement, ast.Expr) and isinstance(
            statement.value, ast.Constant
        ):
            continue
        return False
    return True


def is_interface(node: ast.ClassDef) -> bool:
    """Report whether a class declares an API instead of implementing one.

    Protocols, ABCs, enums and typed containers share no instance state between
    their methods by construction, which LCOM4 would otherwise read as a total
    lack of cohesion.

    Args:
        node: The class definition to inspect.

    Returns:
        True for declaration-only classes.
    """
    if any(base_name(base) in INTERFACE_BASES for base in node.bases):
        return True
    methods = [
        child
        for child in node.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    return bool(methods) and all(is_stub(method) for method in methods)


def compute_lcom4(node: ast.ClassDef) -> tuple[list[list[str]], list[str]]:
    """Compute the LCOM4 components and the method names they were built from.

    Two methods are connected when they share a `self` attribute or one calls
    the other. LCOM4 is the number of connected components of that graph;
    returning the components themselves makes the number actionable, since each
    component is a candidate for its own class.

    Args:
        node: The class definition to inspect.

    Returns:
        A tuple of the components, largest first, and the method names that
        touch no instance state at all.
    """
    definitions = [
        child
        for child in node.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    all_names = {child.name for child in definitions}
    bound = [
        child
        for child in definitions
        if child.name not in EXCLUDED_FROM_COHESION and not is_unbound(child)
    ]

    # A method that never reads `self` trivially forms its own component, which
    # says nothing about the class. Constant-returning properties are the
    # common case. They are reported separately instead of inflating LCOM4.
    references = {child.name: self_references(child) for child in bound}
    stateless = sorted(name for name, used in references.items() if not used)
    considered = [child for child in bound if references[child.name]]
    if not considered:
        return [], stateless

    touched = {child.name: references[child.name] for child in considered}
    parent = {child.name: child.name for child in considered}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    names = list(touched)
    for i, left in enumerate(names):
        # A direct call to a sibling method is an edge on its own.
        for called in touched[left] & all_names:
            if called in parent:
                union(left, called)
        for right in names[i + 1 :]:
            shared = (touched[left] & touched[right]) - all_names
            if shared:
                union(left, right)

    grouped: dict[str, list[str]] = {}
    for name in names:
        grouped.setdefault(find(name), []).append(name)
    components = sorted(grouped.values(), key=len, reverse=True)
    return components, stateless


def collect_classes() -> tuple[dict[str, ClassInfo], dict[str, list[ast.ClassDef]]]:
    """Parse the package and build a ClassInfo per class definition.

    Returns:
        A tuple of the qualified-name to ClassInfo mapping and, per module, the
        raw ClassDef nodes, so a second pass can resolve cross-class references.
    """
    classes: dict[str, ClassInfo] = {}
    nodes_by_module: dict[str, list[ast.ClassDef]] = {}

    for path in source_files():
        module = module_name_for(path)
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            print(f"skipping {module}: {exc}", file=sys.stderr)
            continue

        radon_classes = {
            (block.name, block.lineno): block
            for block in cc_visit(source)
            if isinstance(block, RadonClass)
        }
        definitions = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        nodes_by_module[module] = definitions

        for node in definitions:
            components, stateless = compute_lcom4(node)
            radon_block = radon_classes.get((node.name, node.lineno))
            method_complexities = (
                [method.complexity for method in radon_block.methods]
                if radon_block
                else []
            )
            info = ClassInfo(
                name=node.name,
                module=module,
                lineno=node.lineno,
                bases=[b for b in (base_name(base) for base in node.bases) if b],
                methods=[name for component in components for name in component],
                stateless=stateless,
                wmc=sum(method_complexities),
                max_cc=max(method_complexities, default=0),
                components=components,
                interface=is_interface(node),
            )
            classes[info.qualified] = info

    return classes, nodes_by_module


def compute_coupling(
    classes: dict[str, ClassInfo], nodes_by_module: dict[str, list[ast.ClassDef]]
) -> None:
    """Fill in CBO and NOC for every class.

    CBO counts distinct other project classes named anywhere in the class body.
    Names are matched bare, so two project classes sharing a name are conflated;
    that is accepted here in exchange for not needing to resolve imports.

    Args:
        classes: All measured classes, mutated in place.
        nodes_by_module: Raw ClassDef nodes keyed by module.
    """
    by_bare_name: dict[str, list[str]] = {}
    for qualified, info in classes.items():
        by_bare_name.setdefault(info.name, []).append(qualified)

    for module, definitions in nodes_by_module.items():
        for node in definitions:
            info = classes[f"{module}.{node.name}"]
            referenced: set[str] = set()
            for child in ast.walk(node):
                candidate = None
                if isinstance(child, ast.Name):
                    candidate = child.id
                elif isinstance(child, ast.Attribute):
                    candidate = child.attr
                if candidate and candidate != node.name and candidate in by_bare_name:
                    referenced.update(by_bare_name[candidate])
            info.cbo = len(referenced)

    for info in classes.values():
        for base in info.bases:
            for qualified in by_bare_name.get(base, []):
                classes[qualified].noc += 1


def inheritance_depth(info: ClassInfo, classes: dict[str, ClassInfo]) -> int:
    """Compute the depth of inheritance tree, counting project bases only.

    Args:
        info: The class to measure.
        classes: All measured classes.

    Returns:
        Number of project superclasses above it; 0 for a project-level root.
    """
    by_bare_name: dict[str, ClassInfo] = {}
    for other in classes.values():
        by_bare_name.setdefault(other.name, other)

    depth = 0
    seen = {info.qualified}
    frontier = list(info.bases)
    while frontier:
        base = frontier.pop()
        parent = by_bare_name.get(base)
        if parent is None or parent.qualified in seen:
            continue
        seen.add(parent.qualified)
        depth += 1
        frontier.extend(parent.bases)
    return depth


def verdict(info: ClassInfo, cbo_cut: int) -> str:
    """Classify a class from its cohesion and coupling.

    Args:
        info: The measured class.
        cbo_cut: CBO value at which coupling counts as high.

    Returns:
        One of `interface`, `split candidate`, `god class`, `highly coupled`,
        `ok`.
    """
    if info.interface:
        return "interface"
    incohesive = info.lcom4 >= 2 and len(info.methods) >= 4
    coupled = info.cbo >= cbo_cut
    if incohesive and coupled:
        return "god class"
    if incohesive:
        return "split candidate"
    if coupled:
        return "highly coupled"
    return "ok"


def render(classes: dict[str, ClassInfo]) -> str:
    """Render the class-level markdown report.

    Args:
        classes: All measured classes.

    Returns:
        The report as a markdown string.
    """
    measured = [info for info in classes.values() if not info.interface]
    interfaces = len(classes) - len(measured)
    cbo_values = sorted(info.cbo for info in measured if info.cbo)
    cbo_cut = cbo_values[int(len(cbo_values) * 0.9)] if cbo_values else 0

    ranked = sorted(
        measured,
        key=lambda info: (info.lcom4, info.cbo, info.wmc),
        reverse=True,
    )
    verdicts: dict[str, list[ClassInfo]] = {}
    for info in classes.values():
        verdicts.setdefault(verdict(info, cbo_cut), []).append(info)

    lines = [
        "# Class-level coupling and cohesion (CK metrics)",
        "",
        f"- Classes: {len(classes)} ({interfaces} declaration-only, not measured)",
        f"- Mean LCOM4: {sum(i.lcom4 for i in measured) / max(len(measured), 1):.2f}",
        f"- Mean CBO: {sum(i.cbo for i in measured) / max(len(measured), 1):.2f}",
        f"- High-coupling cut-off (90th percentile): CBO >= {cbo_cut}",
        "",
        "LCOM4 counts connected components of the method graph, so 1 is a",
        "cohesive class and n means n unrelated responsibilities share a",
        "namespace. CBO counts distinct other project classes referenced.",
        "",
        "Excluded from LCOM4, because each would register as an isolated",
        "component regardless of how well the class is designed: `__init__` and",
        "`__post_init__` (they touch every attribute), staticmethods and",
        "classmethods (no `self`), and methods that read no instance state at",
        "all. Protocols, ABCs, enums and stub-only classes are dropped whole.",
        "",
        "## Least cohesive / most coupled classes",
        "",
        "| Class | Module | Methods | LCOM4 | CBO | WMC | Max CC | DIT | NOC | Verdict |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for info in ranked[:25]:
        lines.append(
            f"| `{info.name}` | `{info.module}` | {len(info.methods)} | {info.lcom4} | "
            f"{info.cbo} | {info.wmc} | {info.max_cc} | "
            f"{inheritance_depth(info, classes)} | {info.noc} | {verdict(info, cbo_cut)} |"
        )

    lines += ["", "## Verdicts", ""]
    for key in ("god class", "split candidate", "highly coupled", "ok", "interface"):
        group = verdicts.get(key, [])
        lines.append(f"- **{key}** ({len(group)})")
        if key not in ("ok", "interface"):
            for info in sorted(group, key=lambda i: (-i.lcom4, -i.cbo)):
                lines.append(
                    f"  - `{info.qualified}` (LCOM4 {info.lcom4}, CBO {info.cbo}, "
                    f"WMC {info.wmc})"
                )
                for island in info.islands:
                    members = ", ".join(f"`{name}`" for name in sorted(island))
                    lines.append(f"    - detachable: {members}")

    return "\n".join(lines) + "\n"


def main() -> None:
    """Measure every class, write the report and echo it to stdout."""
    classes, nodes_by_module = collect_classes()
    compute_coupling(classes, nodes_by_module)
    report = render(classes)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"written to {REPORT_PATH.relative_to(REPO_ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
