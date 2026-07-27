# launcher - Arbeitsnotizen

Alles, was dieser Agent probiert, findet und verwirft. Roh und fortlaufend.
Kein Politur-Anspruch, aber jede Zahl mit Quelle (SLURM-Job, Run-Dir, Datei:Zeile).

## Welle 1 - zwei 30-min-Prod-Sonden

### Configs

- `scripts/slurm/configs/duell_l3_hybrid_p2048.yaml`: extends `l3_hybrid`
  (-> `hybrid_v1` -> `_base`). Deep merge (scripts/slurm/launch.py:102
  `_deep_merge`) - args merge per-key, kein voller Parent-Override noetig.
  Uebersteuert nur `args.prefill: 2048`, `args.output_dir`, `args.wandb_name`,
  `job_name: r2d-L3du-hyb`, `output_dir` (Slurm-Log-Verzeichnis),
  `env.SEED: "42"` (verify.sh prueft die YAML-Datei statisch auf `SEED:.*42`,
  nicht nur den Launch-Aufruf - siehe verify.sh:95).
- `scripts/slurm/configs/duell_l3_aggpool_p2048.yaml`: extends
  `l3_aggregator_pooled` (-> `aggregator_pooled_l1` -> `_base`), gleiche
  Uebersteuerungen, `job_name: r2d-L3du-agg`.
- Auffaelligkeit gegen Auftrag: geerbtes `args.steps` ist **1500000**, nicht
  2000000 - `hybrid_v1`/`aggregator_pooled_l1` ueberschreiben den `_base`-
  Default von 2000000 explizit (Quelle: `scripts/slurm/configs/hybrid_v1.yaml`
  Kommentar "Ladder budget ... 1.5M"). Nicht angepasst, da 30 min Walltime das
  Budget ohnehin weit vor 1.5M bzw. 2M Steps kappt - irrelevant fuer die
  30-min-Sonde. Verifiziert per Dry-Run.
- Dry-Run-Render bestaetigt fuer beide: `--prefill 2048`, `--seed "${SEED}"`
  mit `SEED=42` exportiert, korrekter `output_dir`
  (`output/runs/duell-l3-hybrid-p2048/run-${SLURM_JOB_ID}` bzw.
  `output/runs/duell-l3-aggpool-p2048/run-${SLURM_JOB_ID}`).
- `bash prototyp/duell-vggt-integration/verify.sh` -> PASS (nach Ergaenzung
  von `env.SEED: "42"` in beiden neuen YAMLs; vorher FAIL auf Punkt 3 Seed,
  weil verify.sh die YAML-Datei selbst nach `SEED:.*42` grep't statt nur den
  Launch-Befehl zu pruefen).

### Submits

- Vor dem Submit: `squeue -u ul_hfj15` zeigte bereits laufende/pending Jobs
  desselben Users (u.a. `r2d-L3-a` RUNNING auf uc3n082 seit 27:44 min,
  `r2d-L3-p` RUNNING auf uc3n104 seit 27:44 min, plus diverse PENDING) -
  vermutlich Vor-Duell-Baselines, nicht Teil dieser Aufgabe. Nicht angefasst
  (Blindheit/Scope), nur zur Kenntnis genommen.
- Job A (hybrid, p2048): submitted 2026-07-27 11:14:57 CEST.
  `bash scripts/slurm/launch.sh duell_l3_hybrid_p2048 --prod --time 00:30:00 \
    --partition dev_gpu_h100 --exclude uc3n089 --env SEED=42`
  -> jid=6057639
- Job B (aggpool, p2048): submitted 2026-07-27 11:15:04 CEST.
  `bash scripts/slurm/launch.sh duell_l3_aggpool_p2048 --prod --time 00:30:00 \
    --partition gpu_h100_short --env SEED=42`
  -> jid=6057641

(wird fortgesetzt, sobald die Jobs starten/laufen)

### Startzeiten (squeue-Beobachtung)

- Job B (aggpool, 6057641): CF um 11:15:04, RUNNING seit ca. 11:16:03 auf
  Node `uc3n104`, Partition `gpu_h100_short`. Startup-Log zeigt normale
  habitat_sim-Init (GL-Renderer NVIDIA H100/PCIe, ObjectNav-v1 Task) -
  die "SemanticScene ... Aborting load" / "hwloc_set_cpubind failed"-Zeilen
  in slurm-6057641.err sind bekannte harmlose Habitat/NUMA-Warnungen, kein
  Fehler.
- Job A (hybrid, 6057639): initial PENDING mit Reason
  `QOSMaxJobsPerUserLimit` auf Partition `dev_gpu_h100`. Ursache (per
  `sacctmgr show qos dev`): QOS `dev` hat `MaxJobsPU=1` - nur 1 laufender Job
  pro User auf dev_gpu_h100 gleichzeitig, und Slot war durch einen
  vorbestehenden fremden Job (6057422, `r2d-L3-a`, RUNNING seit 11:14 auf
  uc3n082) belegt. Dieser Vorlauf-Job lief bereits ~28-30 min und traf die
  dev-Partitions-Grenze `MaxTime=00:30:00` (Quelle: `scontrol show partition
  dev_gpu_h100`), wodurch der Slot frei wurde. Job A ging danach automatisch
  in RUNNING (kein Resubmit auf gpu_h100_short noetig, kein exit-134-Sterben
  im Prefill - lief innerhalb der ~10-min-Regel an).
- Job A RUNNING seit ca. 11:19:0x auf Node `uc3n082`, Partition
  `dev_gpu_h100`. (Quelle: `squeue -j 6057639` state RUNNING, TIME 0:03 zum
  Beobachtungszeitpunkt kurz nach Start.)

### Erste Fortschritts-Bestaetigung (waehrend des Laufs)

- Job B (aggpool, 6057641): MANIFEST.json bestaetigt `prefill_steps: 2048`,
  `seed: 42`, `total_steps: 1500000`, `adapter: aggregator_pooled`,
  `logdir: output/runs/duell-l3-aggpool-p2048/run-6057641`, `started_at:
  2026-07-27T09:18:56 UTC` (= 11:18:56 CEST), Node `uc3n104.localdomain`
  (Quelle: `output/runs/duell-l3-aggpool-p2048/run-6057641/MANIFEST.json`).
  W&B Run-ID `oop0uehe` (Quelle: slurm-6057641.err, wandb-Setup-Zeilen).
  Erste `metrics.csv`-Zeilen bei step=1 vorhanden (18 Zeilen inkl. Header,
  Quelle: `output/runs/duell-l3-aggpool-p2048/run-6057641/metrics.csv`):
  loss/dyn=9.43, loss/rew=5.54, total_loss=71.85, nan_skipped=0.0.
  `perf/ms_per_step_interval` bei step=1 = 65320.76 ms - das ist NICHT die
  reale Step-Zeit, sondern die kumulierte Prefill-Zeit (2048 Env-Steps)
  geteilt durch die erste Trainings-Step-Zaehlung; die naechste Zeile bei
  step=250 (log_every=250) wird die reale Trainingsgeschwindigkeit zeigen.
  Achtung beim Auswerten: `metrics.csv` ist nicht step-sortiert (bekannt).
- Job A (hybrid, 6057639): MANIFEST.json bestaetigt `prefill_steps: 2048`,
  `seed: 42`, `total_steps: 1500000`, `adapter: rgb_pointmap_pose`,
  `logdir: output/runs/duell-l3-hybrid-p2048/run-6057639`, `started_at:
  2026-07-27T09:21:13 UTC` (= 11:21:13 CEST), Node `uc3n082.localdomain`.
  W&B Run-ID `ovxk4xxn`. `metrics.csv` zum Beobachtungszeitpunkt noch 0
  Zeilen (nur angelegt, erster Log-Eintrag steht noch aus) - GPU-Auslastung
  laut `gpu-memory-6057639.csv` bei 80-90% (aktiv rechnend, kein Hang),
  waehrend Job B's GPU-Log zeitgleich 0% zeigte (Env-Stepping/CPU-gebunden
  waehrend Prefill bei aggpool, kein Fehlerzeichen).
- Auffaelligkeit: die Print-Zeilen "Prefilling N steps..." /
  "Training from step..." (src/r2dreamer/launch/loops.py:372,424)
  erscheinen in keinem der beiden Slurm-Logs, obwohl metrics.csv bereits
  Trainings-Loss-Werte zeigt - vermutlich stdout-Blockbuffering bei
  Datei-Redirect (kein `flush=True`), keine echte Anomalie. Nicht weiter
  verfolgt, da `metrics.csv` die verlaessliche Quelle ist.

## Welle 2 - vorbereitet (Update vom Orchestrator waehrend Welle 1 lief)

Orchestrator-Nachricht erhalten waehrend Welle-1-Jobs liefen: Welle-2-Configs
`scripts/slurm/configs/duell_l3_hybrid_lottery.yaml` und
`duell_l3_aggpool_lottery.yaml` liegen bereits im Worktree (Commit `2536d88`
"feat(duell): lottery knobs + kv-budget-200k pooled arm for the 30-min L3
duel"). Eigene Verifikation statt Blindvertrauen:

- Beide Dateien existieren, `extends: l3_hybrid` bzw.
  `extends: l3_aggregator_pooled`, gleiche `env.SEED: "42"`-Konvention.
- Hybrid-Lottery: `args.prefill: 1024`, `args.train_ratio: 256`,
  `args.act_entropy: 0.1`, output_dir `output/runs/duell-l3-hybrid-lottery/
  run-${SLURM_JOB_ID}`.
- Aggpool-Lottery: gleiche drei Knobs + eigener `run_id:
  habitat-l3-aggregator-pooled-b200k` (KV-Budget 200k, Adapter
  `aggregator_pooled_b200k`) - verifiziert vorhanden in
  `scripts/r2dreamer/_run_configs.py:271-276`.
- Dry-Run beider Configs (`--prod --dry-run --time 00:30:00 --exclude
  uc3n089 --env SEED=42`) sauber: prefill 1024, seed 42, train_ratio 256,
  act_entropy 0.1, korrekte output_dir/run_id, restliche Args unveraendert
  aus dem jeweiligen Parent.
- `bash prototyp/duell-vggt-integration/verify.sh` -> PASS fuer alle 4
  Configs (Welle 1 + Welle 2).

Plan: sobald 6057641 (agg, Welle 1) die Queue verlaesst, sofort
`duell_l3_aggpool_lottery --prod --time 00:30:00 --partition gpu_h100_short
--exclude uc3n089 --env SEED=42` submitten (per Monitor-Task automatisiert,
um den Slot nicht brachliegen zu lassen). Sobald 6057639 (hyb, Welle 1) die
Queue verlaesst, sofort `duell_l3_hybrid_lottery --prod --time 00:30:00
--partition dev_gpu_h100 --exclude uc3n089 --env SEED=42` submitten; falls
dev binnen ~5 min nicht startet oder im Prefill mit exit 134 stirbt, auf
gpu_h100_short resubmitten (Regel aus RULES.md 4 / PROBLEMS.md, wie bei
Welle 1 dev-Job-Wartezeit durch QOSMaxJobsPerUserLimit beobachtet - das war
allerdings kein exit-134-Fall, sondern reine Slot-Wartezeit).

Welle-1-Zahlen werden SOFORT nach Jobende gemeldet (Orchestrator-Anforderung,
Auswertungsschluss ~12:50), nicht erst nach Welle 2.

## Welle 1 - Ergebnis (TIMEOUT = Erfolg)

- Job B (aggpool p2048, 6057641): `sacct -j 6057641` -> State TIMEOUT,
  ExitCode 0:0, Elapsed 00:30:31 (Batch-Schritt CANCELLED 0:15 - erwartetes
  Habitat-GL-Teardown-Muster, siehe MANIFEST-Regel, kein echter Fehler).
  `MANIFEST.json` (output/runs/duell-l3-aggpool-p2048/run-6057641/
  MANIFEST.json): prefill_steps=2048, seed=42, adapter=aggregator_pooled,
  logdir korrekt. metrics.csv: 1188 Zeilen (inkl. Header) - reale Trainings-
  Steps liefen durch. Kein `ended_at`/`status` im mitgeloggten Manifest-Auszug
  gesehen (Monitor-Snapshot lief evtl. vor dem finalen write_manifest_end;
  spaeter beim Kopieren in runs/ pruefen).
- Job A (hybrid p2048, 6057639): `sacct -j 6057639` -> State TIMEOUT,
  ExitCode 0:0, Elapsed 00:30:29, gleiches CANCELLED-0:15-Muster.
  MANIFEST: prefill_steps=2048, seed=42, adapter=rgb_pointmap_pose. 
  metrics.csv: 1058 Zeilen.
- Beide Welle-1-Jobs damit erfolgreich im Sinne der Regeln (TIMEOUT nach
  30 min = Erfolg, kein Absturz, kein exit-134-Muster in den Logs).

## Welle 2 - Auto-Resubmit ausgeloest

- Sobald Job B (6057641) die Queue verliess (11:52:16), automatisch
  `duell_l3_aggpool_lottery --prod --time 00:30:00 --partition
  gpu_h100_short --exclude uc3n089 --env SEED=42` submitted -> jid=6057871.
- Sobald Job A (6057639) die Queue verliess (11:54:17), automatisch
  `duell_l3_hybrid_lottery --prod --time 00:30:00 --partition dev_gpu_h100
  --exclude uc3n089 --env SEED=42` submitted -> jid=6057877 (nach urspruenglichem
  Plan auf dev_gpu_h100, wie im ersten Orchestrator-Update besprochen).

## Anweisung von Luca (ueber Coordinator, 12:47 CEST)

Luca: Welle 2 soll NICHT auf `dev_gpu_h100`, sondern BEIDE Welle-2-Jobs auf
`gpu_h100_short`. Zum Zeitpunkt der Anweisung Status geprueft:

- `duell_l3_aggpool_lottery` (6057871): war bereits auf `gpu_h100_short`
  submitted (Quelle: `sacct -j 6057871` Partition=gpu_h100_short) und zu dem
  Zeitpunkt schon TIMEOUT/fertig (Start 11:52:29, Elapsed 00:30:22) - passt
  ohnehin zur Anweisung, keine Aktion notwendig.
- `duell_l3_hybrid_lottery` (6057877): war zum Zeitpunkt der Anweisung
  bereits RUNNING auf `dev_gpu_h100` (Node uc3n082, Start 12:21:02, Quelle:
  `sacct -j 6057877`), TIME bei Pruefung 26:14 von 30:00 - also fast fertig.
  Gemaess Luca's expliziter Regel ("Laeuft er schon, lass ihn laufen und
  melde es mir") NICHT gecancelt, sondern laufen gelassen. Wird nach Ende
  normal ausgewertet und hier gemeldet. Kein Konflikt mit der Compute-Regel
  (max 2 parallele GPU-Jobs), da zu diesem Zeitpunkt nur dieser eine Job lief.

## Korrektur von Luca (12:4x CEST, ueber Coordinator)

Praezisierung: nichts canceln, egal ob PENDING oder RUNNING auf
dev_gpu_h100 - nur falls noch gar nicht submitted, dann gpu_h100_short
nehmen. Deckt sich mit dem bereits gewaehlten Vorgehen (6057877 lief zum
Zeitpunkt der ersten Anweisung schon auf dev_gpu_h100 und wurde nicht
angefasst). Keine weitere Aktion notwendig, Job laeuft weiter bis TIMEOUT.

## Zahlen (aus metrics.csv, step-sortiert ausgelesen, Python-Skript ueber csv-Datei)

### Job B - duell_l3_aggpool_p2048 (6057641, gpu_h100_short, uc3n104)
- max Step (irgendeine Metrik): 9001
  (Quelle: output/runs/duell-l3-aggpool-p2048/run-6057641/metrics.csv)
- metrics/sr letzter Eintrag: step=8904, sr=0.0556 (5.56 %)
- metrics/spl letzter Eintrag: step=8904, spl=0.0201
- episode/count letzter Eintrag: step=8904, count=18
- episode/steps-Eintraege (Episodenzahl): 18
- perf/ms_per_step_interval letzte 3 Log-Punkte: step 8501=132.7ms,
  8751=137.7ms, 9001=134.1ms (eingeschwungener Zustand)
- sacct Elapsed: 00:30:31 (=1831s), State=TIMEOUT, ExitCode=0:0
- ms/Step nach Formel (Laufzeit_s/Steps*1000): 1831/9001*1000 = 203.4 ms/step
  (inkl. Setup+Prefill-Overhead; eingeschwungen laut perf-Metrik ~133-138ms)

### Job A - duell_l3_hybrid_p2048 (6057639, dev_gpu_h100, uc3n082)
- max Step: 8001
- metrics/sr letzter Eintrag: step=7999, sr=0.0 (0 %)
- metrics/spl letzter Eintrag: step=7999, spl=0.0
- episode/count letzter Eintrag: step=7999, count=16
- episode/steps-Eintraege: 16
- perf/ms_per_step_interval letzte 3: step 7501=151.8ms, 7751=146.0ms,
  8001=152.0ms
- sacct Elapsed: 00:30:29 (=1829s), State=TIMEOUT, ExitCode=0:0
- ms/Step nach Formel: 1829/8001*1000 = 228.6 ms/step (eingeschwungen
  ~146-152ms)

### Job Welle-2 aggpool-lottery (6057871, gpu_h100_short, uc3n104)
- max Step: 19675 (fast 2.5x mehr Steps als p2048-Variante dank prefill=1024
  + train_ratio=256 + KV-Budget 200k)
- metrics/sr letzter Eintrag: step=19675, sr=0.025 (2.5 %)
- metrics/spl letzter Eintrag: step=19675, spl=0.0034
- episode/count / episode-steps-Eintraege: 40
- perf/ms_per_step_interval letzte 3: step 19003=71.7ms, 19251=71.8ms,
  19503=71.4ms (deutlich schneller als p2048-Variante, passt zur
  jianyuan-Schaetzung -20 bis -30ms/step durch KV-Budget + kuerzerer
  train_ratio-Anteil)
- sacct Elapsed: 00:30:22 (=1822s), State=TIMEOUT, ExitCode=0:0
- ms/Step nach Formel: 1822/19675*1000 = 92.6 ms/step

Alle drei MANIFEST.json-Dateien haben `started_at` + vollen config-Block,
aber KEIN `ended_at`/`status` (SLURM TIMEOUT killt den Prozess, bevor
write_manifest_end laufen kann) - laut RULES.md/Aufgabenstellung ist TIMEOUT
selbst das Erfolgskriterium, nicht der Exit-Code oder ein fehlendes
`status`-Feld. metrics.csv waechst durchgaengig, keine Traceback/Error-Zeilen
in den .err-Logs (nur bekannte harmlose Habitat-SemanticScene-Warnungen und
JAX-cuda_timer-Warmup-Hinweise).

Artefakte kopiert nach:
- prototyp/duell-vggt-integration/2026-07-27/runs/6057641-aggpool-p2048/
- prototyp/duell-vggt-integration/2026-07-27/runs/6057639-hybrid-p2048/
- prototyp/duell-vggt-integration/2026-07-27/runs/6057871-aggpool-lottery/
(je: slurm-*.out/.err, metrics.csv, MANIFEST.json, rendered-sbatch.sh)

### Job Welle-2 hybrid-lottery (6057877, dev_gpu_h100, uc3n082, lief auf Anweisung
Lucas dort weiter statt gpu_h100_short)

- sacct: State=TIMEOUT, ExitCode=0:0, Elapsed=00:30:19, Start=12:21:02,
  End=12:51:21 CEST. Erwartetes CANCELLED-DUE-TO-TIME-LIMIT-Muster im
  .err-Log (Zeile 310), keine sonstigen Traceback/Error-Zeilen.
- MANIFEST.json: prefill_steps=1024, train_ratio=256, act_entropy=0.1,
  seed=42, adapter=rgb_pointmap_pose, logdir korrekt,
  git_branch=duell/2026-07-27-lottery-knobs-kv200k, git_sha=2536d88.
- max Step: 9503 (vs. 8001 bei p2048-Baseline mit gleichem Adapter -
  nur +19 %, deutlich weniger Zugewinn als bei aggpool-lottery, weil der
  hybrid-Pfad kein KV-Budget-Cap bekommen hat, nur prefill/train_ratio/
  act_entropy geaendert wurden)
- metrics/sr letzter Eintrag: step=9499, sr=0.0 (0 %)
- metrics/spl letzter Eintrag: step=9499, spl=0.0
- episode/count / episode-steps-Eintraege: 19
- perf/ms_per_step_interval letzte 3: step 9003=139.8ms, 9251=133.1ms,
  9503=137.5ms
- ms/Step nach Formel: 1819/9503*1000 = 191.4 ms/step (Elapsed 00:30:19 =
  1819s)
- Artefakte kopiert nach:
  prototyp/duell-vggt-integration/2026-07-27/runs/6057877-hybrid-lottery/

### Abschluss Welle 1 + Welle 2

- Alle 4 Jobs (6057639, 6057641, 6057871, 6057877) erfolgreich per TIMEOUT
  beendet, keine echten Fehler in den Logs (nur bekannte Habitat/JAX-
  Warmup-Warnungen und das erwartete "CANCELLED DUE TO TIME LIMIT").
- `squeue -u ul_hfj15` zeigt keine verbleibenden L3du-Jobs mehr - Queue
  sauber, keine der 2 GPU-Job-Grenze verletzt.
- Ergebnis-Ueberblick (SR/SPL sind bei so wenigen Episoden/so fruehem Step
  extrem verrauscht, siehe episode/count je Zeile - nicht als belastbare
  Erfolgsquote interpretieren, nur als Sanity-Check dass trainiert wird):

  | Job     | Arm                 | Steps | SR     | SPL    | Episoden | ms/Step (Formel) | ms/Step eingeschwungen |
  |---------|----------------------|-------|--------|--------|----------|-------------------|------------------------|
  | 6057641 | aggpool p2048        | 9001  | 5.56%  | 0.0201 | 18       | 203.4             | ~133-138               |
  | 6057639 | hybrid p2048         | 8001  | 0.0%   | 0.0    | 16       | 228.6             | ~146-152               |
  | 6057871 | aggpool lottery(kv200k)| 19675| 2.5%   | 0.0034 | 40       | 92.6              | ~71-72                 |
  | 6057877 | hybrid lottery       | 9503  | 0.0%   | 0.0    | 19       | 191.4             | ~133-140               |
