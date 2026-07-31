# Regeln - Duell 4: world_points als Injektion

Diese Datei ist bindend. Bei Zweifelsfaellen gilt der Wortlaut hier, nicht die
eigene Einschaetzung. Sie gilt fuer **beide** Seiten gleichermassen.

## 1. Zeit

- **Fuenf Stunden Gesamtzeit**, Wall-Clock. Die Uhr startet fuer beide Seiten
  gleichzeitig mit dem vereinbarten Startsignal (der Startprompt an den
  Agent).
- Es zaehlt **alles**: Nachdenken, Code schreiben, SLURM-Queue-Wartezeit,
  Laufzeit der Jobs, Auswertung.
- **Der letzte gewertete Seed-42-Lauf wird spaetestens bei T+4:00
  abgesetzt.** Was spaeter startet, wird vor Ablauf nicht mehr fertig und ist
  verlorene Queue-Zeit.
- Architektur wird gebaut, waehrend GPUs rechnen. Die Queue steht nie still,
  weil noch Code entsteht.

## 2. Blindheit und Vorwissen

- **Geteiltes Vorwissen:** die Ordner `2026-07-27/`, `2026-07-27-r2/` und
  `2026-07-29-r3/` sind fuer beide Seiten lesend offen und ausdruecklich
  empfohlen.
- **Blind waehrend des Fensters:** Branches, Configs, Notizen, Ledger-Eintraege
  und Zwischenergebnisse der Gegenseite sind bis zum Ablauf tabu. Dass Jobs
  der Gegenseite in `squeue` sichtbar sind, laesst sich nicht vermeiden;
  Job-Namen verraten deshalb keine Konfigurationsdetails.
- Jede Seite arbeitet auf eigenen Branches:
  `duell/2026-07-31-r4-agent-<kurz>` bzw. `duell/2026-07-31-r4-luca-<kurz>`.
- Kein Sync zwischen Wellen. Der Austausch passiert in der Nachbesprechung.

## 3. Eingefrorene Zone (harte Grenze)

### 3.1 Pfade

```
src/environments/**
data/curriculum/**
src/shared/wandb_utils.py
prototyp/duell-vggt-integration/verify.sh
```

Damit sind eingefroren: `GOAL_RADIUS`, die Done-Bedingung,
`max_episode_steps = 500`, das Rolling-100-Fenster der Metrik-Aggregation,
das Curriculum-JSON und die SR-Berechnung. Ein Gate, das man unterwegs
aufbohrt, ist kein Gate.

### 3.2 Trainings-Hyperparameter

Die Trainingsknobs sind **eingefroren**. Jeder gewertete Lauf faehrt die
C/P2-Config:

| Knob | Wert | Herkunft |
|---|---|---|
| Curriculum | `L3` | Duell-Vorgabe |
| Seed | `42` (Bestaetigung `43`, Abschnitt 6) | Duell-Vorgabe |
| `prefill` | `1024` | `duell2_l3_aggpool_b200k_tr128.yaml` |
| `train_ratio` | `128` | dito |
| `act_entropy` | `0.1` | dito |
| KV `total_budget` | `200_000` | Default seit `43f5a1c` |
| `compute_heads` | **`True`** (Duell-Vorgabe, weicht von r3 ab) | `GOAL.md`, harte Bedingung |
| `batch_size` / `seq_len` | `16` / `64` | `_base.yaml` |
| Lernrate | Default | `_base.yaml` |

### 3.3 Messung

Die Definition der Messung, die Wertungsmatrix und die paarweise
Seed-Zuordnung stehen in `GOAL.md` und sind eingefroren. Der Scorer wird vor
Welle 1 aus `2026-07-29-r3/agents/orchestrator/score.py` abgeleitet, auf die
P2-Referenzen gestellt und gegen P2 selbst validiert (~0.00); danach gilt er
als Teil der Messung und wird nicht mehr angefasst.

## 4. Freies Spielfeld

Alles andere ist erlaubt und erwuenscht: Adapter (`src/adapters/**`),
Encoder-Module (`src/r2dreamer/**`, `src/vggt/**`), neue YAMLs unter
`scripts/slurm/configs/`. Neue Varianten folgen dem Muster aus
`src/r2dreamer/AGENTS.md:44-56`.

### 4.1 Harte Bedingung

Siehe `GOAL.md`: `world_points` in jedem gewerteten Arm, `compute_heads=True`,
**keine Aggregator-Tokens**, RGB-Bild als einzige erlaubte Ergaenzung.

### 4.2 Speicherdeckel

`buffer_capacity` ist frei, aber die **Vorbelegung ist auf 32 GB gedeckelt**:

```
buffer_capacity * Zeilengroesse <= 32 GB
```

Jeder Arm traegt **Zeilengroesse und Kapazitaet ins Ledger**. `mem: 64G` wird
nicht hochgesetzt (verlaengert die Queue-Wartezeit). Zur Orientierung:

| Beispiel-Arm | Zeile | max. Kapazitaet |
|---|---|---|
| gepoolte Point-Map 37x37x3 fp32 (~16 KB) + Pose | ~16 KB | 500 000 (= 8 GB) |
| dichte Point-Map 518x518x3 fp16 | ~1.6 MB | **~20 000 Zeilen** |
| dichte Point-Map 518x518x3 fp32 | ~3.2 MB | **~10 000 Zeilen** |

Ein dichter Arm bekommt also ein winziges Replay-Fenster und zahlt zusaetzlich
ueber `ms/Step`. Das ist die ehrliche Kostenwahrheit, keine Abschreckung.

## 5. Compute

- Gewertete Laeufe: **ausschliesslich `gpu_h100_short`**.
  - **`dev_gpu_h100` ist exklusiv fuer Luca.** Fuer den Agent tabu.
  - `uc3n089` immer per `--exclude` ausschliessen.
- **Slot-Pool: maximal 4 parallele GPU-Jobs insgesamt, beide Seiten
  zusammen.** Wer zuerst kommt, mahlt zuerst. Keine Reservierung.
- Jobs um ~1 Minute gestaffelt absetzen (uv-sync-Race aus r2).
- Ein gewerteter Lauf:

  ```bash
  bash scripts/slurm/launch.sh <variante> --prod --time 00:30:00 \
      --partition gpu_h100_short --exclude uc3n089 --env SEED=42
  ```

- `--smoke` ist fuer gewertete Laeufe **verboten** (Step-Cap 1500, eigene
  JAX-Speicheroptionen). Fuer Syntax- und Startchecks bleibt es erlaubt.
- GPU-Code niemals auf dem Login-Node, immer ueber `srun`/`sbatch`.

## 6. Verifikation und der zweite Seed

- Vor dem Ende des Fensters laeuft fuer jede Seite
  `bash prototyp/duell-vggt-integration/verify.sh`. Schlaegt es fehl, sind
  die Laeufe dieser Seite ungueltig.
- **Seed 42 laeuft im Fenster.** Nach Ablauf bestaetigt jede Seite ihren
  besten Arm auf **Seed 43** - dieselbe Config, Seed per CLI:

  ```bash
  bash scripts/slurm/launch.sh <variante> --prod --time 00:30:00 \
      --partition gpu_h100_short --exclude uc3n089 --env SEED=43
  ```

  Der CLI-Override ist Pflicht: `verify.sh` verlangt von jeder geaenderten
  YAML ein literales `SEED: "42"`.
- Der Seed-43-Lauf wird **gegen 6089423** gewertet, der Seed-42-Lauf gegen
  6087075.

## 7. Wertung und Abschluss

- **Sieger = hoeheres Score-Mittel** (Seed 42 + Seed 43) des besten Arms pro
  Seite. Abstand unter ~0.05 wird als innerhalb der Ziehungsvarianz benannt.
- **Kein PR aus dem Duell.** Merges laufen spaeter ueber den normalen Weg.
- Nach den Bestaetigungslaeufen baut der Agent die **interaktive
  Ergebnis-Tabelle (Widget)** ueber alle Laeufe beider Seiten (alle Metriken
  der Matrix plus sr, N, Zeilengroesse/Kapazitaet, Job-Ids), danach folgt die
  gemeinsame Nachbesprechung; deren Fazit kommt in die Konsolidierung des
  Ledgers.

## 8. Wohin alles geschrieben wird

Alles bleibt in `prototyp/duell-vggt-integration/2026-07-31-r4/`. Nichts geht
nach `docs/notes/`.

- Agent-Seite: jeder Subagent schreibt nur in
  `2026-07-31-r4/agents/<sein-name>/NOTES.md`; das zentrale `LEDGER.md`
  pflegt der Orchestrator (Abschnitt "Agent").
- **Luca-Seite: `2026-07-31-r4/agents/luca/NOTES.md`** - mindestens Arm,
  Job-Id, Score pro Lauf. Lehre aus Duell 1: ohne Mensch-Ledger bleibt die
  Konsolidierung leer.
- Logs, `metrics.csv` und gerenderte sbatch-Skripte nach
  `2026-07-31-r4/runs/<jobid>-<arm>/`.
- Jede Zahl braucht eine Quelle: SLURM-Job-Id, Run-Verzeichnis, W&B-Id oder
  `datei.py:zeile`.

## 9. Besetzung (Agent-Seite)

Wie in r3: Orchestrator, Launcher, Analyst, Hypothesist, dazu die Personas
`danijar-hafner` (RL, Replay, Entropie) und `jianyuan-wang` (VGGT, Heads,
Kostenzerlegung). Wang ist hier besonders einschlaegig: Point-Head-Kosten,
Scale-Free-Koordinaten und die Box-Mean-Kritik (Punkte im leeren Raum an
Tiefenkanten) stammen aus seinen Notizen.

## 10. Infrastruktur-Rauschen

Ein abgebrochener Job ist nicht automatisch ein Bug der eigenen Aenderung.
Regel wie in r3: einmal neu absetzen (Retry-Budget 1 pro Arm); bricht er
erneut ab, gilt der Arm als `gescheitert` und der Fehler wird im Ledger
vermerkt statt debuggt.
