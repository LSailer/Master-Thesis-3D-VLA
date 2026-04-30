# Plan: Claude Code Automation Upgrades

> Source PRD: #67

## Architectural decisions

- **W&B MCP**: Local npx server, reuses existing `WANDB_API_KEY` from user settings
- **Hook pattern**: PreToolUse exit-code convention (0=allow, 2=block), consistent with existing `block-dangerous-git.sh`
- **Subagent**: Standalone `.claude/agents/` file, read-only tools, manual invocation only
- **Skill**: User-invocable only (`disable-model-invocation: true`), generates `.sbatch` from existing templates

---

## Phase 1: All four automation upgrades

### What to build

1. **Block external/ edits hook**
   - Create `.claude/hooks/block-external-edits.sh` (PreToolUse, blocks Edit/Write to paths containing `/external/`)
   - Wire into `.claude/settings.json` PreToolUse hooks array
   - Follow same JSON-parsing pattern as existing `block-dangerous-git.sh`

2. **W&B MCP server**
   - Add `wandb-mcp-server` entry to `.mcp.json` using npx transport
   - No auth config needed — server reads `WANDB_API_KEY` from environment

3. **Shape-checker subagent**
   - Create `.claude/agents/shape-checker.md`
   - Read-only tools: Read, Grep, Glob
   - Instructions: trace tensor shapes through forward passes, check batch dimension consistency, cross-reference with config files
   - Key files to reference: `modules/dreamerv3/networks.py`, `modules/r2dreamer/networks.py`, config files

4. **`/experiment` skill**
   - Create `.claude/skills/experiment/SKILL.md`
   - User-invocable only (`disable-model-invocation: true`)
   - Workflow: read wiki experiment page for context → find most similar existing `.sbatch` template → generate new script → show user for confirmation → submit with `sbatch` → report job ID
   - Include partition reference table (dev_gpu_h100, gpu_h100, gpu_h100_il)
   - W&B naming convention: `--wandb_name "<experiment>-${SLURM_JOB_ID}"`, project `3d-vla-objectnav`
   - Common patterns from existing scripts: `uv run python`, seed from `$SLURM_JOB_ID`, 8 CPUs, 64G mem, output to `output/<type>/slurm-%j.out`

### Acceptance criteria

- [ ] Editing a file in `external/` is blocked with a clear error message
- [ ] W&B MCP server responds to queries in Claude Code
- [ ] Shape-checker agent can be invoked and produces a shape trace for `modules/r2dreamer/networks.py`
- [ ] `/experiment test-run` generates a valid `.sbatch` script matching existing template patterns
- [ ] `/experiment` submits the generated script with `sbatch` after user confirmation
