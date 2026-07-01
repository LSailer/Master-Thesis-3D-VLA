# UI Prototype

Generate **several radically different UI variations** in a scratchpad-only prototype, switchable from a floating bottom bar. The user flips between variants, picks one (or steals bits from each), then throws the rest away.

If the question is about logic/state rather than what something looks like — wrong branch. Use [LOGIC.md](LOGIC.md).

## When this is the right shape

- "What should this page look like?"
- "I want to see a few options for this dashboard before committing."
- "Try a different layout for the settings screen."
- Any time the user would otherwise spend a day picking between three vague mockups in their head.

## Scratchpad-only shape

UI prototypes live under `scratchpad/prototypes/<prototype-slug>/` or an active `scratchpad/experiments/<experiment-slug>/prototype/`. Read the existing page/component for context, but do not edit it during prototype work.

### Shape A — mirror an existing page or section (preferred)

Use this when the design would eventually live inside an existing route or component. Create a scratchpad prototype that mirrors the relevant page density, data shape, and constraints. Import read-only helpers/components only when that is simple and safe; otherwise copy minimal mock data and state the assumptions in the prototype `README.md`.

This is preferred over editing the real route because it preserves prototype freedom while keeping the design grounded in real context.

### Shape B — standalone scratchpad surface

Use this only when the thing being prototyped has no existing page to mirror. Create a standalone scratchpad app, HTML file, notebook, or script following the project's available runtime. Name the folder so it is obviously a prototype.

Before choosing Shape B, sanity-check: is there really no existing page or workflow this should mirror? An empty standalone surface hides design problems that a populated one would expose.

In both shapes the floating switcher is scratchpad-local.

## Process

### 1. State the question and pick N

Default to **3 variants**. More than 5 stops being radically different and starts being noise — cap there.

Write down the plan in the prototype `README.md`:

> "Three scratchpad variants of the settings page, switchable via `?variant=`, mirroring the existing `/settings` route without editing production files."

This works whether the user is here to push back or not.

### 2. Generate radically different variants

Draft each variant. Hold each one to:

- The page's purpose and the data it has access to.
- The project's component library / styling system where practical.
- A clear exported component/function name, e.g. `VariantA`, `VariantB`, `VariantC`.

Variants must be **structurally different** — different layout, different information hierarchy, different primary affordance, not just different colours. Three slightly-tweaked card grids isn't a UI prototype, it's wallpaper. If two drafts come out too similar, redo one with explicit "do not use a card grid" guidance.

### 3. Wire them together

Create a scratchpad-local switcher entrypoint:

```tsx
// pseudo-code — adapt to the scratchpad runtime
const variant = searchParams.get('variant') ?? 'A';
return (
  <>
    {variant === 'A' && <VariantA {...data} />}
    {variant === 'B' && <VariantB {...data} />}
    {variant === 'C' && <VariantC {...data} />}
    <PrototypeSwitcher variants={['A','B','C']} current={variant} />
  </>
);
```

For Shape A: mirror the existing route's data and layout constraints in scratchpad; only the rendered prototype subtree changes per variant.

For Shape B: the scratchpad prototype owns its own minimal mock data and route/file.

### 4. Build the floating switcher

A small fixed-position bar at the bottom-centre of the screen with three pieces:

- **Left arrow** — cycles to the previous variant (wraps around).
- **Variant label** — shows the current variant key and, if the variant exports a name, that name too. e.g. `B — Sidebar layout`.
- **Right arrow** — cycles forward (wraps around).

Behaviour:

- Clicking an arrow updates the scratchpad URL/search param or local state so the variant is reload-stable when the runtime supports it.
- Keyboard: `←` and `→` arrow keys also cycle. Don't intercept arrow keys when an `<input>`, `<textarea>`, or `[contenteditable]` is focused.
- Visually distinct from the page (e.g. high-contrast pill, subtle shadow) so it's obviously not part of the design being evaluated.
- Keep it inside the scratchpad prototype. If the winning design is later folded into production, do not include the switcher unless the user explicitly wants a production feature flag.

### 5. Hand it over

Surface the run command and the `?variant=` keys if supported. The user will flip through whenever they get to it. The interesting feedback is usually **"I want the header from B with the sidebar from C"** — that's the actual design they want.

### 6. Capture the answer and clean up

Once a variant has won, write down which one and why in the scratchpad `NOTES.md`, an issue comment, an ADR, or the eventual production commit message. Then:

- Delete losing scratchpad variants when they are no longer useful.
- Propose the winning design as a separate production patch/snippet, or ask for explicit production-edit approval.
- Delete the scratchpad switcher when the prototype is absorbed.

Don't leave variant components or the switcher lying around. They rot fast and confuse the next reader.

## Anti-patterns

- **Variants that differ only in colour or copy.** That's a tweak, not a prototype. Real variants disagree about structure.
- **Sharing too much code between variants.** A shared `<Header>` is fine; a shared `<Layout>` defeats the point. Each variant should be free to throw out the layout.
- **Wiring variants to real mutations.** Read-only prototypes are fine. If a variant needs to mutate, point it at a stub — the question is "what should this look like", not "does the backend work".
- **Editing production routes for prototype work.** The prototype skill is scratchpad-only. Production integration is a separate approved change.
- **Promoting the prototype directly to production.** The variant code was written under prototype constraints (no tests, minimal error handling). Rewrite it properly when you fold it in.
