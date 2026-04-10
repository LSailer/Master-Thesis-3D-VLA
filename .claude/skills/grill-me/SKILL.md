---
name: grill-me
description: Interview the user relentlessly about a plan or design until reaching shared understanding, resolving each branch of the decision tree. Uses web search to back recommendations with real-world patterns, papers, and known pitfalls. Use when user wants to stress-test a plan, get grilled on their design, or mentions "grill me".
---

Interview me relentlessly about every aspect of this plan until we reach a shared understanding. Walk down each branch of the design tree, resolving dependencies between decisions one-by-one. For each question, provide your recommended answer.

If a question can be answered by exploring the codebase, explore the codebase instead.

## Web-backed research

When grilling, proactively use WebSearch to strengthen your questions and recommendations:

- **Uncertainty**: When you are unsure about best practices or trade-offs for a design decision, search the web before asking or recommending. Do not guess — look it up.
- **Common patterns**: Search for established patterns, conventions, and prior art relevant to each branch of the decision tree. Cite what you find.
- **Pitfalls and failure modes**: Search for known issues, gotchas, or failure modes that others have encountered with the approach under discussion.
- **Research context**: For academic/ML topics, search for relevant papers, benchmarks, or comparisons that inform the decision.

When you find relevant information, briefly summarize the source and how it applies. Do not dump raw search results — integrate findings into your questions and recommendations naturally.

## Session conclusion

Once all branches of the decision tree are resolved and shared understanding is reached, ask the user how they want to capture the outcome:

1. **Write a plan document** — Save the full summary (goal, success criteria, key decisions, deliverables, implementation order) as a markdown file in `docs/` and commit it. This is the default recommendation for multi-step plans.
2. **Start implementing directly** — Skip the doc and proceed to implementation immediately (e.g., via `/engineer`).

Do not end the session without asking. Always present both options.
