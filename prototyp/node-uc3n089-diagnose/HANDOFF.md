# Handoff

2026-07-30: Folder created. Evidence gathered from the 2026-07-30 CNN smoke
waves (dead: 6098260-6098263 on uc3n089; green resubmit: 6098413-6098416 on
uc3n088) is summarized in GOAL.md. probe.sbatch + submit.sh written, not yet
run. Next: `./submit.sh uc3n089` and `./submit.sh uc3n088` (control), then
compare `outputs/prototype/node-uc3n089-diagnose/<node>-<jobid>/summary.txt`
and the per-stage .err files. uc3n089 is excluded by default in
scripts/slurm/configs/_base.yaml - remove that once the node is fixed.
