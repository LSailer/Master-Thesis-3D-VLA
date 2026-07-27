# Duell-Durchlauf 2026-07-27, Runde 2

Alles, was dieser Duell-Durchlauf erzeugt, liegt in diesem Ordner. Nichts davon
geht nach `docs/notes/`.

Regeln: `../RULES.md`. Ziel und Wertungsmatrix: `../GOAL.md`.
Der erste Durchlauf liegt in `../2026-07-27/` und ist lesend offen.

## Aufbau

```
2026-07-27-r2/
  README.md            diese Datei
  LEDGER.md            das zentrale Ergebnis-Ledger (Orchestrator pflegt es)
  agents/
    orchestrator/      Entscheidungen, Zeitplan, was wann losgeschickt wurde
    hypothesist/       Kandidaten, Begruendungen, verworfene Ideen
    launcher/          Configs, sbatch-Aufrufe, Job-Ids, Fehlschlaege
    analyst/           Auswertungen, Zahlen, Ableseprotokolle
    danijar-hafner/    Beratung World Model / Dreamer
    jianyuan-wang/     Beratung VGGT / 3D-Features
  runs/                Logs, kopierte metrics.csv, sbatch-Renderings
```

## Regeln fuer das Schreiben

- **Jeder Agent schreibt ausschliesslich in seinen eigenen Ordner.** Fremde
  Agent-Ordner sind lesend offen, schreibend tabu.
- **Lesen ist ausdruecklich erwuenscht.** Wer eine Idee hat, schaut vorher in
  `agents/hypothesist/NOTES.md`, in `LEDGER.md` und in das Ledger des ersten
  Durchlaufs, ob sie schon tot ist.
- **`LEDGER.md` pflegt nur der Orchestrator.** Subagents liefern Zahlen, tragen
  sie aber nicht selbst ein.
- **Jede Zahl mit Quelle**: SLURM-Job-Id, Run-Verzeichnis, W&B-Id oder
  `datei.py:zeile`. Eine Zahl ohne Quelle ist keine Zahl.
- **Rohes ist besser als poliertes.** Diese Notizen sind Arbeitsmaterial, kein
  Bericht. Sackgassen aufschreiben ist genauso wertvoll wie Erfolge.
- Logs und `metrics.csv` nach `runs/` kopieren, damit die Zahlen den Lauf
  ueberleben.

## Abweichung von der Prototyp-Konvention

`prototyp/CLAUDE.md` sagt normalerweise: keine generierten Outputs im
Prototyp-Ordner, die gehoeren nach `outputs/prototype/<feature>/`. Fuer das
Duell gilt bewusst das Gegenteil: Logs, Auswertungen und Findings bleiben hier,
damit alles zum Durchlauf an einem Ort liegt und die Agents sich gegenseitig
lesen koennen.
