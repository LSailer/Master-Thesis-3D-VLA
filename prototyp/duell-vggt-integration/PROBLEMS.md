# Bekannte Probleme und Stolperfallen

Stand vor Duellbeginn, aus der Codebasis verifiziert. Waehrend des Duells
fortschreiben.

## Blocker, die Zeit kosten koennen

1. **Keine 3D-Arm-Run-Id fuer L3.** `scripts/r2dreamer/_run_configs.py` hat
   Nicht-`rgb`-Adapter nur fuer L1. Eine L3-Variante muss erst angelegt werden
   (RUN_CONFIGS-Eintrag + YAML unter `scripts/slurm/configs/`).

2. **Seed ist aktuell die SLURM-Job-Id.** `scripts/slurm/configs/l3_cnn.yaml:19`
   setzt `seed: ${SLURM_JOB_ID}`. Fuer das Duell muss `seed: ${SEED}` mit
   `env: {SEED: "42"}` gesetzt werden. Neue Configs von Anfang an so anlegen.

3. **`--curriculum_path` ist im r2dreamer-Pfad tot.** Es wird geparst
   (`parser.py:58-62`), aber nirgends weitergereicht.
   `resolve_habitat_curriculum_path` akzeptiert nur die Namen `L1..L4`
   (`src/environments/habitat.py:68-78`). Nur `src/baselines/random_agent.py`
   nutzt einen echten Pfad. Also: mit `--curriculum L3` arbeiten, nicht mit
   einem Pfad.

4. **Alte Checkpoints laden nicht auf HEAD.** Der Adapter-Refactor hat
   Flax-Modulpfade umbenannt. Ein Warmstart aus einem bestehenden L3-Checkpoint
   funktioniert nicht. Jeder Lauf startet von Null.

5. **`curriculum_check`-Guard zeigt ins Leere.** `l3_cnn.yaml:5` rendert einen
   Guard, der `scripts/environments/generate_curriculum.py` aufruft. Dieser Pfad
   existiert nicht (nur `__archiv__/environments/generate_curriculum.py`). Wenn
   das L3-JSON auf dem Cluster fehlt, laeuft der Job trotzdem an und stirbt
   spaeter beim Oeffnen der Datei (`habitat.py:182`).

6. **`l3_cnn.yaml` setzt kein `buffer_capacity`.** Es laeuft damit auf dem
   Default 500k, waehrend die L1-Headline-Zahl mit 1M lief. Bei einem
   30-Minuten-Lauf irrelevant, aber beim Vergleich mit alten Zahlen relevant.

7. **`perf/ms_per_step_interval` gibt es in keinem historischen L3-Lauf.** Die
   Keys kamen erst nach allen L3-Jobs dazu. Am HEAD werden sie geloggt, alte
   Zahlen sind nicht vergleichbar.

## Cluster-Stolperfallen

- **Login-Node.** Training, Eval, Habitat, VGGT und Smoke-Tests niemals dort
  ausfuehren. Immer `srun` oder `sbatch`.
- **`uc3n089`** bricht bei habitat GL-Reads ab. Per `--exclude` ausschliessen.
- **`SBATCH_EXCLUDE` wird von Slurm 25.11 ignoriert**, deshalb `--exclude` als
  Flag an `launch.sh` uebergeben.
- **`dev_gpu_h100`**: laut `scripts/slurm/README.md:127-132` bricht dort der
  OpenGL-Renderer beim Prefill ab. Luca hat es zuletzt erfolgreich verwendet.
  Wenn ein dev-Job im Prefill stirbt: sofort auf `gpu_h100_short` wechseln,
  nicht debuggen.
- **Queue-Wartezeit.** Es gibt einen belegten Fall mit ueber zwei Stunden
  `PENDING` ohne Allokation. Waehrend ein Job wartet, am naechsten Kandidaten
  weiterarbeiten, nicht blockieren.
- **W&B ist auf Compute-Nodes gesperrt.** Smoke-Jobs exportieren
  `WANDB_MODE=offline`. Die `metrics.csv` im Run-Verzeichnis enthaelt alles,
  was W&B auch bekaeme. Bei Bedarf vom Login-Node aus
  `wandb sync <output_dir>/wandb/offline-run-*`.
- **`scripts/setup_worktree.sh` nutzt GNU-`realpath --relative-to`.** Auf dem
  Cluster in Ordnung, auf macOS nicht. In frischen Worktrees vor dem ersten Lauf
  ausfuehren, sonst fehlen `data`, `.venv`, `output` und `external`.

## Messfalle

`val_every: 0` in allen Prod-Configs bedeutet: alle berichteten `metrics/sr`
sind **Trainings**-Episoden unter dem stochastischen Actor, keine Held-out-
Metrik. Das gilt fuer Baseline und Duell-Laeufe gleichermassen, der Vergleich
bleibt also fair. Es darf nur nicht als Generalisierungsleistung gelesen
werden.
