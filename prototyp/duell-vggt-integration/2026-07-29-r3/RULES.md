# Regeln - Duell 3: Frame-Tokens und Kamera-Tokens

Diese Datei ist bindend. Bei Zweifelsfaellen gilt der Wortlaut hier, nicht die
eigene Einschaetzung.

## 1. Zeit

- **Drei Stunden Gesamtzeit**, Wall-Clock, ab dem ersten Tool-Call.
- Es zaehlt **alles**: Nachdenken, Code schreiben, SLURM-Queue-Wartezeit,
  Laufzeit der Jobs, Auswertung, PR-Erstellung.
- Nach drei Stunden ist Schluss. Was dann nicht ausgewertet ist, zaehlt nicht.
- **Die letzte Welle wird spaetestens bei T+1:45 abgesetzt.** Darin liegen der
  Bestaetigungslauf auf Seed 43 und der Kontrolllauf. Was danach noch startet,
  wird nicht mehr fertig und ist verlorene Queue-Zeit.
- **Architektur wird nur gebaut, waehrend GPUs rechnen.** Die Queue steht nie
  still, weil noch Code entsteht. Einzige Ausnahme ist P1: der muss vor
  Welle 1 existieren.

## 2. Eingefrorene Zone (harte Grenze)

### 2.1 Pfade

Folgende Pfade duerfen **nicht veraendert** werden:

```
src/environments/**
data/curriculum/**
src/shared/wandb_utils.py
prototyp/duell-vggt-integration/verify.sh
```

Damit sind eingefroren: `GOAL_RADIUS`, `_is_success_distance`, die
Done-Bedingung, `max_episode_steps = 500`, das Rolling-100-Fenster der
Metrik-Aggregation, das Curriculum-JSON und die SR-Berechnung in
`EpisodeTracker`. `verify.sh` steht mit auf der Liste: ein Gate, das man
unterwegs aufbohrt, ist kein Gate.

### 2.2 Trainings-Hyperparameter

Anders als in Duell 2 sind die Trainingsknobs **eingefroren**. Jeder gewertete
Lauf faehrt Cs Config:

| Knob | Wert | Herkunft |
|---|---|---|
| Curriculum | `L3` | Duell-Vorgabe |
| Seed | `42` (Bestaetigung `43`, siehe Abschnitt 7) | Duell-Vorgabe |
| `prefill` | `1024` | `duell2_l3_aggpool_b200k_tr128.yaml` |
| `train_ratio` | `128` | dito |
| `act_entropy` | `0.1` | dito |
| KV `total_budget` | `200_000` | `AggregatorPooledAdapter.EXTRACTOR_KWARGS` |
| `compute_heads` | `False` | dito |
| `batch_size` / `seq_len` | `16` / `64` | `_base.yaml` |
| Lernrate | Default | `_base.yaml` |

`aggregator_pooled_b200k` ist seit Commit `43f5a1c` nur noch ein **Alias** von
`aggregator_pooled` - das 200k-Budget ist der Default. Ein Adapter, der von
`AggregatorPooledAdapter` erbt, bekommt ihn automatisch.

**Der Camera-Head bleibt aus.** "Kamera-Tokens" meint Token 0 der Frame-Haelfte
und Token 0 der globalen Haelfte, beide latent und gratis. Der echte
Camera-Head (`camera_pose`) braeuchte `compute_heads=True` und kostet ms/Step;
der Geometrie-Arm hat in r2 mit -0.5829 verloren.

### 2.3 Messung

Die Definition der Messung, die Wertungsmatrix und die paarweise
Seed-Zuordnung (siehe `GOAL.md`).

## 3. Freies Spielfeld

Alles andere ist erlaubt und ausdruecklich erwuenscht:

- **Der Adapter.** Neue Adapter-Module, Zusammensetzung des Beobachtungs-
  vektors, Pooling-Verfahren, Aufteilung in mehrere `AdapterField`s,
  `src/adapters/**`.
- **Der Encoder.** Eigene Encoder-Module unter `src/r2dreamer/encoders/`, neue
  `Encoder`-Enum-Mitglieder, Routing in `routed_composite.py`, `mlp_hidden` /
  `mlp_layers`, `src/r2dreamer/**`, `src/vggt/**`.
- Neue Run-Ids und neue YAMLs unter `scripts/slurm/configs/`.

Neue Varianten folgen dem Muster aus `src/r2dreamer/AGENTS.md:44-56`: Adapter +
`ADAPTERS`-Registrierung + Encoder-Routing + Branch unter `encoders/` +
Verdrahtung in `routed_composite.py` + `RUN_CONFIGS`-Eintrag + SLURM-YAML.
Kein neuer Python-Shim. Eine Variante, die sich nur in einer Konstante
unterscheidet, ist eine Subklasse, keine Kopie der Pipeline.

### 3.1 Harte Bedingung: die Frame-Haelfte

**Jeder gewertete Arm enthaelt die Frame-Haelfte der Aggregator-Tokens.** Ein
Arm, der nur die globale Haelfte anders poolt oder kodiert, beantwortet die
Frage dieses Duells nicht und wird nicht gewertet.

### 3.2 Speicherdeckel

`buffer_capacity` ist frei, aber die **Vorbelegung ist auf 32 GB gedeckelt**:

```
buffer_capacity * Zeilengroesse <= 32 GB
```

Jeder Arm traegt **Zeilengroesse und Kapazitaet ins Ledger**.

Gemessene Grundlage: SLURM fordert `mem: 64G` pro Job (`_base.yaml:14`), und
der C-Lauf 6060404 hat davon 13.20 GB benutzt
(`slurm-6060404.out:104`). 64 - 13 Grundlast - 32 Replay laesst rund 19 GB
Puffer. `mem` wird **nicht** hochgesetzt: ein groesserer Request verlaengert die
Queue-Wartezeit, und die ist im Drei-Stunden-Fenster teurer als der Speicher.

Was der Deckel praktisch bedeutet:

| Arm | Zeile | max. Kapazitaet |
|---|---|---|
| C (Referenz) | 3072 fp32 = 12 KB | 500 000 (= 6 GB), unveraendert |
| P1 | 6144 fp32 = 24 KB | 500 000 (= 12 GB) |
| P2 | 4096 fp32 = 16 KB | 500 000 (= 8 GB) |
| volle Sequenz | 1374 x 2048 fp16 = 5.6 MB | **~5 700 Zeilen** |

Ein Sequenz-Arm bekommt also ein Replay-Fenster von ~5 700 Schritten, waehrend
ein gepoolter Arm bei 500 000 bleibt, und zahlt zusaetzlich ueber `ms/Step`.
Das ist die ehrliche Kostenwahrheit der Sequenz-Arme, keine Abschreckung: wer
sie fahren will, faehrt sie und schreibt das Ergebnis ins Ledger.

## 4. Compute

- Partition: **ausschliesslich `gpu_h100_short`**.
  - **`dev_gpu_h100` ist tabu.** Die Partition bleibt Luca vorbehalten.
  - `uc3n089` immer per `--exclude` ausschliessen (bricht bei habitat GL-Reads ab).
- **Maximal 4 parallele GPU-Jobs.**
- Ein gewerteter Lauf ist **30 Minuten Walltime im `--prod`-Modus**:

  ```bash
  bash scripts/slurm/launch.sh <variante> --prod --time 00:30:00 \
      --partition gpu_h100_short --exclude uc3n089 --env SEED=42
  ```

  `--smoke` ist fuer gewertete Laeufe **verboten**. Es deckelt bei 1500 Steps
  statt bei 30 Minuten und setzt eigene JAX-Speicheroptionen
  (`launch.py:236-243`), die das Laufzeitverhalten veraendern. Fuer schnelle
  Syntax- und Startchecks bleibt `--smoke` erlaubt; ein Absturz dort sagt
  nichts ueber die Aenderung aus.
- GPU-Code niemals auf dem Login-Node ausfuehren, immer ueber `srun` bzw.
  `sbatch`.
- **Jobs um ~1 Minute gestaffelt absetzen.** In r2 hat ein uv-sync-Race
  parallel startender Jobs aus demselben Worktree die geteilte `.venv`
  zerlegt und Slot B gekostet (6060403, exit 2 nach 31 s).

## 5. Wellenplan

| Welle | spaetestens | Slots |
|---|---|---|
| **1** | T+0:45 | **P1 gesetzt** (mit Cs MLP-Encoder, unveraendert) + drei Arme nach Wahl |
| **2** | T+1:45 | Seed-43-Bestaetigung des Fuehrenden, **Kontrolllauf C auf Seed 42**, zwei Arme aus den Erkenntnissen von Welle 1 |

Der Kontrolllauf ist eine frische Ziehung von Cs unveraenderter Config. Er
wird nicht gewertet; er misst, wie viel des beobachteten Abstands blosse
Ziehungsvarianz ist (r2: ~+/-0.04 Score).

## 6. Berichtspflicht

Keine Gegenseite, keine Blindheit. Nach jeder Welle meldet der Agent Luca in
drei Zeilen: was lief, welche Scores kamen heraus, was als naechstes losgeht.

## 7. Verifikation und der zweite Seed

Vor jedem PR und am Ende des Duells laeuft
`bash prototyp/duell-vggt-integration/verify.sh`. Das Script prueft die
eingefrorene Zone, die Curriculum-Pruefsumme und den Seed. Schlaegt es fehl,
ist der Lauf ungueltig.

**Der Bestaetigungslauf nutzt dieselbe Config wie der Gewinner** und setzt den
Seed ueber die Kommandozeile:

```bash
bash scripts/slurm/launch.sh <variante> --prod --time 00:30:00 \
    --partition gpu_h100_short --exclude uc3n089 --env SEED=43
```

Der Grund ist mechanisch: `verify.sh:95` verlangt von **jeder** geaenderten
oder neuen SLURM-Config ein literales `SEED: "42"`. Eine eigene YAML mit
`SEED: "43"` wuerde das Gate reissen, nachdem der Lauf schon 30 Minuten
gerechnet hat. Der CLI-Override ist in `_base.yaml:27` vorgesehen.

Der Seed-43-Lauf wird **gegen 6061173** gewertet, nicht gegen 6060404
(`GOAL.md`, Abschnitt "Referenz").

## 8. Branch und Pull Request

- Branch: `duell/2026-07-29-r3-<kurzbeschreibung>`.
- Ein PR wird **nur geoeffnet, wenn der Score der Schwelle genuegt**:
  Mittelwert aus Seed 42 und Seed 43 **>= +0.10**, gegen C gewertet.
- Es gibt **genau einen PR**, den des bestaetigten Besten. Alle anderen Arme
  ueber der Schwelle stehen nur als Ledger-Zeile mit dem Vermerk
  "ueber Schwelle, unbestaetigt".
- Der Agent oeffnet den PR. **Gemerged wird ausschliesslich von Luca.**
- Niemals auf `main` pushen, niemals force-pushen.
- Kein Agent-Name als Co-Author in Commit-Messages (`CLAUDE.md:6`).

### PR-Body (Pflichtformat)

```markdown
## Ergebnis
- Referenz:                     6060404 (s42) / 6061173 (s43), Duell-2-Sieger C
- Score Seed 42 / Seed 43 / Mittel:   +X.XX / +X.XX / +X.XX
- Schwelle:                     +0.10
- Headline (P1 vs C):           +X.XX
- SLURM Job-Ids:                XXXXXXX (s42), XXXXXXX (s43)
- W&B Run-Ids:                  xxxxxxxx, xxxxxxxx
- Replay-Zeile / Kapazitaet:    XX KB / XXX XXX (= XX GB)

| Metrik   | Ref s42 | Seed 42 | Ref s43 | Seed 43 | Gewicht | Beitrag |
|----------|---------|---------|---------|---------|---------|---------|
| Treffer  | 1       |         | 1       |         | 0.45    |         |
| softspl  | 0.0866  |         | 0.0539  |         | 0.15    |         |
| dtg      | 6.379   |         | 4.975   |         | 0.15    |         |
| spl      | 0.0119  |         | 0.0062  |         | 0.10    |         |
| ms/Step  | 66.8    |         | 69.1    |         | 0.10    |         |
| Episoden | 44      |         | 41      |         | 0.05    |         |
| (sr)     | 0.0227  |         | 0.0244  |         | Bericht |         |

## Was geaendert wurde
<knappe Zusammenfassung des Diffs>

## Warum das den Score verbessert
<Begruendung, keine Spekulation ohne Zahl>

## Verifikation
- `bash prototyp/duell-vggt-integration/verify.sh` : PASS
- Was bewusst nicht getan wurde: <...>
```

## 9. Wohin alles geschrieben wird

Saemtliche Logs, Notizen, Zwischenstaende und Findings gehen in den
Durchlauf-Ordner `prototyp/duell-vggt-integration/2026-07-29-r3/`.
**Nichts davon geht nach `docs/notes/`.**

- Jeder Agent schreibt ausschliesslich in `<ordner>/agents/<sein-name>/NOTES.md`.
  Fremde Agent-Ordner sind lesend offen, schreibend tabu.
- Das zentrale `<ordner>/LEDGER.md` pflegt nur der Orchestrator.
- Logs, `metrics.csv` und gerenderte sbatch-Skripte nach `<ordner>/runs/`
  kopieren.
- Jede Zahl braucht eine Quelle: SLURM-Job-Id, Run-Verzeichnis, W&B-Id oder
  `datei.py:zeile`.
- Die Ordner der ersten beiden Duelle, `2026-07-27/` und `2026-07-27-r2/`,
  sind lesend offen und ausdruecklich empfohlen. `2026-07-27-r2/LEDGER.md`
  listet sechs ausgewertete Laeufe, sieben Erkenntnisse und die offenen
  Faeden, aus denen P3 stammt.

## 10. Besetzung

Wie in r2: Orchestrator, Launcher, Analyst, Hypothesist, dazu die Personas
`danijar-hafner` (RL, Actor-Entropie, Replay) und `jianyuan-wang` (VGGT,
Token-Layout, Kostenzerlegung). Letzterer ist hier besonders einschlaegig: das
Token-Layout und der Pose-Delta-Faden P3 stammen aus seinen r2-Notizen.

## 11. Nach Ablauf

Der Agent uebergibt `2026-07-29-r3/LEDGER.md` mit allen Score-Zeilen, dem
Headline-Vergleich P1 gegen C, der internen Rangliste gegen P1, den Sackgassen
und den offenen Faeden. Ob und was davon spaeter in die Thesis-Dokumentation
wandert, entscheidet Luca danach.
