# Review Rules (gate)

## Primary task: intent check
Read the documents under prototyp/ that belong to this change.
They describe the intended design. Verify: does the diff
implement exactly that? Every deviation is a finding with a
reference to the prototyp/ location.

## Slop patterns (finding when violated)
1. Over-explaining: comments that state WHAT the code does
   rather than WHY.
2. Defensive overkill: excessive try/except blocks or
   redundant type checks.
3. Hallucinated dependencies: importing packages that do
   not exist.
4. Over-engineering: single-purpose helpers that are called
   exactly once.
5. Copy-paste clones: repeating similar logic instead of
   using existing abstractions.

## Risk assessment (in every review)
risk: low | medium | high, with reasoning. Inputs:
diff size (src/ counts fully), diff-cover result,
your own judgment. high => findings are blocking.