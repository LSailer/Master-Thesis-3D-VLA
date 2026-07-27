# Regeln - Duell 2 VGGT-Integration

Diese Datei ist bindend. Bei Zweifelsfaellen gilt der Wortlaut hier, nicht die
eigene Einschaetzung.

## 1. Zeit

- **Drei Stunden Gesamtzeit**, Wall-Clock, ab dem ersten Tool-Call.
- Es zaehlt **alles**: Nachdenken, Code schreiben, SLURM-Queue-Wartezeit,
  Laufzeit der Jobs, Auswertung, PR-Erstellung.
- Nach drei Stunden ist Schluss. Was dann nicht ausgewertet ist, zaehlt nicht.
- **Die letzte Welle wird spaetestens bei T+1:45 abgesetzt.** Darin liegt der
  Bestaetigungslauf auf Seed 43. Was danach noch startet, wird nicht mehr
  fertig und ist verlorene Queue-Zeit.
- **Architektur wird nur gebaut, waehrend GPUs rechnen.** Die Queue steht nie
  still, weil noch Code entsteht.

## 2. Eingefrorene Zone (harte Grenze)

Folgende Pfade duerfen **nicht veraendert** werden:

```
src/environments/**
data/curriculum/**
src/shared/wandb_utils.py
```

Damit sind eingefroren: `GOAL_RADIUS`, `_is_success_distance`, die
Done-Bedingung, `max_episode_steps = 500`, das Rolling-100-Fenster der
Metrik-Aggregation, das Curriculum-JSON und die SR-Berechnung in
`EpisodeTracker`.

`src/shared/wandb_utils.py` steht bewusst mit auf der Liste: dort sitzt die
Aggregation der Success Rate. Ohne diese Datei in der Tabu-Zone liesse sich die
Erfolgsmessung umschreiben, ohne `src/environments/` zu beruehren.

Zusaetzlich eingefroren:

- **Seed = 42** in jedem gewerteten Lauf. Einzige Ausnahme ist der
  Bestaetigungslauf, siehe Abschnitt 7.
- Die Definition der Messung und die Wertungsmatrix (siehe `GOAL.md`).

## 3. Freies Spielfeld

Alles andere ist erlaubt und ausdruecklich erwuenscht:

- **Neue Adapter-Funktionen**, neue Encoder, Encoder-Parameter, Routing,
  `src/vggt/**`, `src/r2dreamer/**`. Das ist der eigentliche Hebel dieses
  Duells, nicht eine Nebenerlaubnis.
- Hyperparameter (lr, batch size, seq_len, act_entropy, buffer capacity, ...)
- Replay-Verhalten, Prefill-Groesse
- **Reward-Shaping.** Der Reward gehoert nicht zur eingefrorenen Zone. Er ist
  im Gegenzug auch aus der Wertung raus (`GOAL.md`).
- SLURM-Configs, Aufloesung, Compute-dtype, Performance-Optimierung
- Neue Run-Ids in `scripts/r2dreamer/_run_configs.py` und neue YAMLs unter
  `scripts/slurm/configs/`

Neue Varianten folgen dem Muster aus `src/r2dreamer/AGENTS.md:44-56`: Adapter +
`ADAPTERS`-Registrierung + Encoder-Routing + Branch unter `encoders/` +
Verdrahtung in `routed_composite.py` + `RUN_CONFIGS`-Eintrag + SLURM-YAML.
Kein neuer Python-Shim (`scripts/r2dreamer/AGENTS.md:18-20`). Eine Variante,
die sich nur in einer Konstante unterscheidet, ist eine Subklasse, keine Kopie
der Pipeline.

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
  statt bei 30 Minuten und setzt zusaetzlich eigene JAX-Speicheroptionen
  (`launch.py:236-243`), die das Laufzeitverhalten veraendern. Fuer schnelle
  Syntax- und Startchecks bleibt `--smoke` erlaubt; ein Absturz dort sagt
  nichts ueber die Aenderung aus.
- GPU-Code niemals auf dem Login-Node ausfuehren, immer ueber `srun` bzw.
  `sbatch`.

## 5. Berichtspflicht

In diesem Durchlauf gibt es **keine Gegenseite und keine Blindheit**. Der Agent
laeuft allein. Nach jeder Welle meldet er Luca in drei Zeilen: was lief, welche
Scores kamen heraus, was als naechstes losgeht.

## 6. Branch und Pull Request

- Branch: `duell/<YYYY-MM-DD>-r2-<kurzbeschreibung>`, zum Beispiel
  `duell/2026-07-27-r2-aggregator-mlp`.
- Ein PR wird **nur geoeffnet, wenn der Score der Schwelle genuegt**:
  Mittelwert aus Seed 42 und Seed 43 **>= +0.10**.
- Es gibt **genau einen PR**, den des bestaetigten Besten. Alle anderen Arme
  ueber der Schwelle stehen nur als Ledger-Zeile mit dem Vermerk
  "ueber Schwelle, unbestaetigt".
- Der Agent oeffnet den PR. **Gemerged wird ausschliesslich von Luca.**
- Niemals auf `main` pushen, niemals force-pushen.
- Kein Agent-Name als Co-Author in Commit-Messages (`CLAUDE.md:6`).

### PR-Body (Pflichtformat)

```markdown
## Ergebnis
- Referenz:                     6057641 (agg-pooled, 30 min, Seed 42)
- Score Seed 42 / Seed 43 / Mittel:   +X.XX / +X.XX / +X.XX
- Schwelle:                     +0.10
- SLURM Job-Ids:                XXXXXXX (s42), XXXXXXX (s43)
- W&B Run-Ids:                  xxxxxxxx, xxxxxxxx

| Metrik   | Referenz | Seed 42 | Seed 43 | Gewicht | Beitrag |
|----------|----------|---------|---------|---------|---------|
| Treffer  | 1        |         |         | 0.45    |         |
| softspl  | 0.0605   |         |         | 0.15    |         |
| dtg      | 5.193    |         |         | 0.15    |         |
| spl      | 0.0201   |         |         | 0.10    |         |
| ms/Step  | 134.1    |         |         | 0.10    |         |
| Episoden | 18       |         |         | 0.05    |         |
| (sr)     | 0.0556   |         |         | Bericht |         |

## Was geaendert wurde
<knappe Zusammenfassung des Diffs>

## Warum das den Score verbessert
<Begruendung, keine Spekulation ohne Zahl>

## Verifikation
- `bash prototyp/duell-vggt-integration/verify.sh` : PASS
- Was bewusst nicht getan wurde: <...>
```

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
`verify.sh` wird waehrend des Duells **nicht angepasst**. Ein Gate, das man
unterwegs aufbohrt, ist kein Gate.

## 8. Wohin alles geschrieben wird

Saemtliche Logs, Notizen, Zwischenstaende und Findings gehen in den
Durchlauf-Ordner `prototyp/duell-vggt-integration/2026-07-27-r2/`.
**Nichts davon geht nach `docs/notes/`.**

- Jeder Agent schreibt ausschliesslich in `<ordner>/agents/<sein-name>/NOTES.md`.
  Fremde Agent-Ordner sind lesend offen, schreibend tabu.
- Das zentrale `<ordner>/LEDGER.md` pflegt nur der Orchestrator.
- Logs, `metrics.csv` und gerenderte sbatch-Skripte nach `<ordner>/runs/`
  kopieren.
- Jede Zahl braucht eine Quelle: SLURM-Job-Id, Run-Verzeichnis, W&B-Id oder
  `datei.py:zeile`.
- Der Ordner des ersten Duells, `2026-07-27/`, ist lesend offen und
  ausdruecklich empfohlen: `LEDGER.md` dort listet vier ausgewertete Laeufe,
  drei Sackgassen und fuenf offene Faeden.

## 9. Nach Ablauf

Der Agent uebergibt `2026-07-27-r2/LEDGER.md` mit allen Score-Zeilen, den
Sackgassen und den offenen Faeden. Ob und was davon spaeter in die
Thesis-Dokumentation wandert, entscheidet Luca danach.
