# Regeln - Duell VGGT-Integration

Diese Datei ist bindend. Bei Zweifelsfaellen gilt der Wortlaut hier, nicht die
eigene Einschaetzung.

## 1. Zeit

- **Zwei Stunden Gesamtzeit**, Wall-Clock, ab dem ersten Tool-Call.
- Es zaehlt **alles**: Nachdenken, Code schreiben, SLURM-Queue-Wartezeit,
  Laufzeit der Jobs, Auswertung, PR-Erstellung.
- Nach zwei Stunden ist Schluss. Was dann nicht ausgewertet ist, zaehlt nicht.

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

- **Seed = 42** in jedem gewerteten Lauf.
- Die Definition der Messung (siehe `GOAL.md`, Erfolgskriterien).

## 3. Freies Spielfeld

Alles andere ist erlaubt und ausdruecklich erwuenscht:

- Encoder, Adapter, Routing, `src/vggt/**`, `src/r2dreamer/**`
- Hyperparameter (lr, batch size, seq_len, act_entropy, buffer capacity, ...)
- Replay-Verhalten, Prefill-Groesse
- Reward-Shaping (der Reward gehoert **nicht** zur eingefrorenen Zone)
- SLURM-Configs, Aufloesung, Compute-dtype, Performance-Optimierung
- Neue Run-Ids in `scripts/r2dreamer/_run_configs.py` und neue YAMLs unter
  `scripts/slurm/configs/`

## 4. Compute

- Partition: **`dev_gpu_h100`** als Primaerwahl, **`gpu_h100_short`** als
  Fallback.
  - Warnung: `scripts/slurm/README.md:127-132` notiert, dass auf `uc3n082`
    (dev) der OpenGL-Renderer von habitat_sim beim Prefill abbricht. Luca hat
    dev zuletzt erfolgreich verwendet. Wenn ein dev-Job im Prefill stirbt,
    sofort auf `gpu_h100_short` wechseln und nicht debuggen.
  - `uc3n089` immer per `--exclude` ausschliessen (bricht bei habitat GL-Reads ab).
- **Maximal 2 parallele GPU-Jobs** pro Seite. Die Queue nicht zustellen.
- Ein gewerteter Lauf ist **30 Minuten Walltime im `--prod`-Modus**:

  ```bash
  bash scripts/slurm/launch.sh <variante> --prod --time 00:30:00 \
      --partition gpu_h100_short --env SEED=42
  ```

  `--smoke` ist fuer gewertete Laeufe **verboten**. Es deckelt bei 1500 Steps
  statt bei 30 Minuten und setzt zusaetzlich eigene JAX-Speicheroptionen
  (`launch.py:236-243`), die das Laufzeitverhalten veraendern. Fuer schnelle
  Syntax- und Startchecks bleibt `--smoke` erlaubt; ein Absturz dort sagt
  nichts ueber die Aenderung aus.
- GPU-Code niemals auf dem Login-Node ausfuehren, immer ueber `srun` bzw.
  `sbatch`.

## 5. Blindheit

Waehrend der zwei Stunden erfaehrt keine Seite, was die andere tut. Kein Blick
in die Branches, PRs, W&B-Runs oder Logs der Gegenseite. Der Austausch findet
ausschliesslich nach Ablauf statt.

## 6. Branch und Pull Request

- Branch: `duell/<YYYY-MM-DD>-<kurzbeschreibung>`, zum Beispiel
  `duell/2026-07-27-aggregator-mlp-l3`.
- Ein PR wird **nur geoeffnet, wenn die Baseline geschlagen wurde**.
- Der Agent oeffnet den PR. **Gemerged wird ausschliesslich von Luca.**
- Niemals auf `main` pushen, niemals force-pushen.
- Kein Agent-Name als Co-Author in Commit-Messages.

### PR-Body (Pflichtformat)

```markdown
## Ergebnis
- Baseline SR (CNN, bei Step N):   X.XX %
- Erreichte SR (dieser Arm):       X.XX %
- Delta:                           +X.XX pp
- N (verglichene Step-Zahl):       XXXXX
- episode/steps (Mittel letzte 100): XXX.X
- perf/ms_per_step_interval:       XXX.X ms
- W&B Run-ID:                      xxxxxxxx
- SLURM Job-ID:                    XXXXXXX
- Seed:                            42

## Was geaendert wurde
<knappe Zusammenfassung des Diffs>

## Warum das die SR verbessert
<Begruendung, keine Spekulation ohne Zahl>

## Verifikation
- `bash prototyp/duell-vggt-integration/verify.sh` : PASS
- Was bewusst nicht getan wurde: <...>
```

## 7. Verifikation

Vor jedem PR und am Ende des Duells laeuft
`bash prototyp/duell-vggt-integration/verify.sh`. Das Script prueft die
eingefrorene Zone, die Curriculum-Pruefsumme und den Seed. Schlaegt es fehl,
ist der Lauf ungueltig.

## 8. Wohin alles geschrieben wird

Saemtliche Logs, Notizen, Zwischenstaende und Findings gehen in den
Durchlauf-Ordner `prototyp/duell-vggt-integration/<datum>/`. **Nichts davon
geht nach `docs/notes/`.**

- Jeder Agent schreibt ausschliesslich in `<datum>/agents/<sein-name>/NOTES.md`.
  Fremde Agent-Ordner sind lesend offen, schreibend tabu.
- Das zentrale `<datum>/LEDGER.md` pflegt nur der Orchestrator.
- Logs, `metrics.csv` und gerenderte sbatch-Skripte nach `<datum>/runs/`
  kopieren.
- Jede Zahl braucht eine Quelle: SLURM-Job-Id, Run-Verzeichnis, W&B-Id oder
  `datei.py:zeile`.

Der Sinn: die Agents sollen sich gegenseitig lesen koennen und wissen, welche
Idee schon tot ist, ohne dass der Orchestrator alles einzeln weiterreichen
muss.

## 9. Nach Ablauf

Beide Seiten treffen sich und tauschen ihre Ledger aus. Der Agent bringt
`<datum>/LEDGER.md` mit, Luca seine Notizen. Die Konsolidierung wird als
gemeinsames Fazit unten in dasselbe `LEDGER.md` geschrieben. Ob und was davon
spaeter in die Thesis-Dokumentation wandert, entscheidet Luca danach.
