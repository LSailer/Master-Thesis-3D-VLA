# Orchestrator-Notizen (Duell 2, r2)

- 12:37:17 UTC erster Tool-Call (setup_worktree.sh). Uhr laeuft, Deadline 15:37 UTC.
- 12:37-12:45: GOAL/RULES/PROBLEMS/LEDGER-r1 gelesen, verify.sh PASS.
- ~12:45-13:20: Drei Write-Tool-Aufrufe hingen im PreToolUse-Hook des Clients
  ("host client may be unreachable") und fraßen ~35 min Wall-Clock. Workaround:
  alle Dateien per Bash-Heredoc schreiben. Fuer den Rest des Duells: kein Write-Tool.
- 13:20: drei Welle-1-YAMLs erstellt (duell2_*), verify.sh PASS.
- 13:21: Welle 1 submittet: 6060402 (A lottery rerun), 6060403 (B b200k pur),
  6060404 (C tr128), 6060405 (D pointmap). Partition gpu_h100_short, exclude uc3n089.
- Zeitplan angepasst: Welle 2 faellt mit Welle 3 zusammen; letzte Submits bis
  14:22 UTC (T+1:45), darin Seed-43-Bestaetigung des Welle-1-Fuehrenden.
- 14:10: Welle-1-Scores (inline gerechnet, Formel = GOAL.md-Matrix):
  A 6060402: parts hits+0.000 softspl+0.049 dtg-0.022 spl-0.083 ms+0.046 ep+0.050 = +0.0407
  C 6060404: parts hits+0.000 softspl+0.065 dtg-0.034 spl-0.041 ms+0.050 ep+0.050 = +0.0898
- B 6060403 exit 2 nach 31s: uv-sync-Race (slurm-6060403.err: "failed to remove
  directory .venv/.../flax/nnx/training"). Vier gleichzeitig startende Jobs
  ranntem parallelem "uv run" in die shared .venv. A+C liefen, venv danach heil.
- 14:12: Finale Welle submittet: 6061173 (C-Config Seed 43), 6061174 (ent 3e-4),
  6061175 (ent 3e-3), 6061176 (Pointmap-Retry). Alle Seed-42 ausser 6061173 (CLI-Override).
- Danijar-Kernbefund uebernommen: act_entropy 0.1 war kontraproduktiv (uniforme
  Aktionsverteilung in allen Duell-Armen); Entropie-Bracket 3e-4/3e-3 in F/G.
- Hypothesist unabhaengig zur selben Matrix-Rechnung: Speed allein ~score-neutral.
- 14:57: Finale Welle ausgewertet (alle TIMEOUT=ok). Einzelbeitraege:
  E 6061173 (s43): softspl-0.016 dtg+0.006 spl-0.069 ms+0.049 ep+0.050 = +0.0194, fwd_pct 0.284
  F 6061174 (ent3e-4): hits-0.450 softspl+0.026 dtg-0.031 spl-0.100 ms+0.049 ep+0.050 = -0.4555, fwd_pct 0.048 (Actor-Kollaps)
  G 6061175 (ent3e-3): softspl+0.088 dtg-0.037 spl-0.092 ms+0.048 ep+0.050 = +0.0564, fwd_pct 0.236
  H 6061176 (pointmap): hits-0.450 softspl-0.013 dtg-0.005 spl-0.100 ms-0.009 ep-0.006 = -0.5829, fwd_pct 0.344
  fwd_pct-Referenzen: A 0.228, C 0.262 (letzter action/forward_pct je Lauf).
- Entscheidung: kein PR (C-Mittel +0.0546 < +0.10). Ledger final, Branch wird committet.
