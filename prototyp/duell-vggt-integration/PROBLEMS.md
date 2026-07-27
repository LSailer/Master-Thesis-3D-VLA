# Bekannte Probleme und Stolperfallen

Stand vor Duellbeginn, aus der Codebasis verifiziert. Waehrend des Duells
fortschreiben.

## Bereits erledigt (Commit `4738326`, 2026-07-26)

Diese drei Punkte waren beim Entwurf des Duells noch Blocker und sind
inzwischen von Luca behoben. Sie stehen hier, damit niemand sie erneut
"repariert".

- **L3-Rungs fuer alle Arme existieren.** `_run_configs.py` fuehrt jetzt
  `habitat-l3-{cnn,hybrid,global-tokens,aggregator-pooled,pointmap-pose}`,
  dazu die YAMLs `l3_{cnn,hybrid,global_tokens,aggregator_pooled,pointmap_pose}.yaml`.
  Fuenf Arme x vier Level, die sich nur im Curriculum unterscheiden.
- **Seed ist reproduzierbar.** Alle Curriculum-Configs nutzen `seed: ${SEED}`
  gegen den Default `SEED: "1"` in `_base.yaml:30`. `--env SEED=42` greift
  damit. Fuer das Duell gilt weiterhin **42**.
- **`buffer_capacity` ist ueberall explizit gesetzt.** Nicht uniform ueber die
  Arme, und das ist Absicht: eine Global-Token-Zeile ist 2.8 MB, dieser Arm
  bleibt bei 20000 gedeckelt.

## Blocker, die Zeit kosten koennen

1. **`--curriculum_path` ist im r2dreamer-Pfad tot.** Es wird geparst
   (`parser.py:58-62`), aber nirgends weitergereicht.
   `resolve_habitat_curriculum_path` akzeptiert nur die Namen `L1..L4`
   (`src/environments/habitat.py:68-78`). Nur `src/baselines/random_agent.py`
   nutzt einen echten Pfad. Also: mit `--curriculum L3` arbeiten, nicht mit
   einem Pfad.

2. **Alte Checkpoints laden nicht auf HEAD.** Der Adapter-Refactor hat
   Flax-Modulpfade umbenannt. Ein Warmstart aus einem bestehenden L3-Checkpoint
   funktioniert nicht. Jeder Lauf startet von Null.

3. **`curriculum_check`-Guard zeigt ins Leere.** Die L3-Configs setzen
   `curriculum_check: data/curriculum/level3_10houses_1goal.json`; der daraus
   gerenderte Guard ruft `scripts/environments/generate_curriculum.py` auf.
   Dieser Pfad existiert nicht (nur `__archiv__/environments/generate_curriculum.py`).
   Wenn das L3-JSON auf dem Cluster fehlt, laeuft der Job trotzdem an und stirbt
   spaeter beim Oeffnen der Datei (`habitat.py:182`).

4. **`perf/ms_per_step_interval` gibt es in keinem historischen L3-Lauf.** Die
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
