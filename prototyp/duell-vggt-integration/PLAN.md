# Plan - wie der Agent das Duell fahren soll

## Rolle

Der Agent ist **Orchestrator**, kein Einzelarbeiter. Er haelt den eigenen
Kontext schlank und sauber, delegiert Ausfuehrung an Subagents und behaelt
selbst nur das Ledger und die Entscheidung, was als naechstes laeuft.

- Effort: **xhigh** fuer den Orchestrator.
- Parallelitaet ist der Hebel gegen die Uhr: mehrere Denker gleichzeitig
  losschicken, nicht sequentiell arbeiten.

## Subagent-Rollen

| Rolle | Aufgabe | Effort | Schreibt nach |
|---|---|---|---|
| `hypothesist` | Kandidaten fuer die Integration vorschlagen, mit Begruendung | high | `<datum>/agents/hypothesist/NOTES.md` |
| `launcher` | Config schreiben, `sbatch` absetzen, Job babysitten, Logs holen | medium | `<datum>/agents/launcher/NOTES.md` |
| `analyst` | `metrics.csv` / W&B auswerten, die Zahlen zurueckgeben | medium | `<datum>/agents/analyst/NOTES.md` |

Jeder Subagent bekommt seinen Ordnerpfad im Prompt mit und den Auftrag, dort
fortlaufend mitzuschreiben. Vor einer neuen Idee liest er `LEDGER.md` und die
Notizen der anderen, damit nichts doppelt probiert wird.

## Personas als Sparringspartner

Zwei Personas stehen als Subagents zur Verfuegung (liegen unter
`~/.claude/agents/` auf dem Cluster, ausserhalb jedes Repos):

- **`danijar-hafner`** - Autor der Dreamer-Reihe. Zustaendig fuer alles, was
  World Model, Latent Dynamics, Replay und Trainingsdynamik betrifft. Modus
  `brainstorm` fuer Design, `dev` fuer Implementierung.
- **`jianyuan-wang`** - Erstautor von VGGT. Zustaendig fuer die 3D-Seite: wie
  VGGT-Features aussehen, welche Tokens welche Information tragen, wie sie in
  den Encoder gehoeren.

Erwarteter Ablauf pro Iteration: Der Orchestrator hat eine Idee, laesst beide
Personas parallel darauf schauen, verdichtet deren Antworten zu einer
Entscheidung und schickt dann den `launcher` los. Die Personas entscheiden
nicht, sie beraten.

## Ablauf

1. **Orientierung (max. 15 min).** Aktuelle Encoder-Varianten und ihr Routing
   verstehen. `src/adapters/`, `src/r2dreamer/encoders/`, `routed_composite.py`,
   `scripts/r2dreamer/_run_configs.py`, `scripts/slurm/configs/`.
2. **Hypothesen (parallel).** `hypothesist` plus beide Personas gleichzeitig.
   Ergebnis: eine geordnete Liste von Kandidaten mit erwarteter Wirkung und
   Implementierungsaufwand.
3. **Erste Welle sofort starten.** Nicht auf perfekte Analyse warten. Die Queue
   ist der Flaschenhals, nicht das Denken. Zwei Kandidaten parallel absetzen,
   waehrend an der naechsten Iteration gearbeitet wird.
4. **Auswerten, eintragen, nachlegen.** Jede fertige Auswertung wandert sofort
   als Zeile nach `<datum>/LEDGER.md`, Logs und `metrics.csv` nach
   `<datum>/runs/`.
5. **PR nur bei Verbesserung.** Format siehe `RULES.md`.

## Messprotokoll (bindend)

1. Der eigene 3D-Lauf endet nach 30 Minuten bei Step `N`.
2. `N` aus der eigenen `metrics.csv` bzw. dem SLURM-Log ablesen.
3. Aus der **Baseline-`metrics.csv`** (CNN, Seed 42, von Luca vor dem Duell auf
   `main` gestartet) die `metrics/sr` **bei Step N** ablesen. Bei fehlendem
   exakten Step den naechstgelegenen geloggten Step nehmen und das im PR
   vermerken.
4. Eigene SR = `metrics/sr` am Ende des eigenen Laufs.
5. Sekundaerwerte: `episode/steps` (Mittel letzte 100) und
   `perf/ms_per_step_interval`.

Beide Zahlen stehen als Long-Format in `<output_dir>/metrics.csv`
(`step,metric,value`). Das funktioniert auch bei `WANDB_MODE=offline`.

## Bekanntes Startproblem

**Es gibt derzeit keine 3D-Arm-Run-Id fuer L3.** `_run_configs.py` fuehrt
Nicht-`rgb`-Adapter ausschliesslich fuer L1. Der erste konkrete Schritt ist
deshalb, eine L3-Variante eines 3D-Arms anzulegen: ein Eintrag in
`RUN_CONFIGS` plus eine YAML unter `scripts/slurm/configs/`. Kein neuer
Python-Shim (`scripts/r2dreamer/AGENTS.md:18-20`).

## Kandidaten fuer die Integration (Startpunkte, nicht abschliessend)

Vorhandene Varianten auf L1, die auf L3 portiert werden koennen:
Aggregator-Pooled, Aggregator-MLP, Global Tokens, Pointmap + Pose,
World-Points/Camera-Pose (WP/CP), Hybrid CNN+VGGT, FiLM-Konditionierung,
House-Context / GNN.

Aus der Laufzeitanalyse: Aggregator-MLP ist mit 38.1k Steps/h der schnellste
3D-Arm, Aggregator-raw mit 6.3k Steps/h der langsamste und fuer ein
30-Minuten-Budget ungeeignet. Bei einem zeitlich gedeckelten Lauf zaehlt
Durchsatz doppelt.

## Arbeitsregeln

- Repo-Regeln aus `CLAUDE.md`, `AGENTS.md`, `REVIEW_RULES.md` und den
  paketweiten `AGENTS.md` gelten unveraendert weiter.
- Kein Em-Dash in Code und Text, nur einfacher Bindestrich.
- Architektur kommt aus dem Adapter-Routing, nicht aus Config-Strings.
- Eine Variante, die sich nur in einer Konstante unterscheidet, ist eine
  Subklasse, keine Kopie der Pipeline.
- Bei drei aufeinanderfolgenden Fehlschlaegen an derselben Sache: abbrechen,
  in `PROBLEMS.md` eintragen, naechster Kandidat.
