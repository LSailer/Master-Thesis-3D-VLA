#!/bin/bash
# Usage: eval_wave.sh "<jobid>:<rundir>:<slot>:<seed> ..."
# Copies metrics.csv + logs into the duel runs/ folder and scores each run.
set -u
cd /pfs/data6/home/ul/ul_student/ul_hfj15/Master-Thesis-3D-VLA/.claude/worktrees/vggt-integration-duel-3-9b748f
SCORE=/scratch/claude-979142/-pfs-data6-home-ul-ul-student-ul-hfj15-Master-Thesis-3D-VLA--claude-worktrees-vggt-integration-duel-3-9b748f/4db47a01-7613-428d-90b8-fb6d8984c556/scratchpad/score.py
DEST=prototyp/duell-vggt-integration/2026-07-29-r3/runs
for spec in "$@"; do
  IFS=: read -r jid rundir slot seed <<< "$spec"
  src="output/runs/$rundir/run-$jid"
  out="$DEST/$jid-$slot"
  mkdir -p "$out"
  cp "$src/metrics.csv" "$out/" 2>/dev/null || echo "$slot: NO metrics.csv in $src"
  cp "output/runs/$rundir/slurm-$jid.out" "$out/" 2>/dev/null
  cp "output/runs/$rundir/slurm-$jid.err" "$out/" 2>/dev/null
  if [ -f "$out/metrics.csv" ]; then
    echo "=== $slot (job $jid, seed $seed) ==="
    python3 "$SCORE" "$out/metrics.csv" "$seed"
  fi
done
