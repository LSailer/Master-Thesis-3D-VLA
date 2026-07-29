# Duell 3 - 2026-07-29-r3

Dritter Durchlauf des VGGT-Integrationsduells. Frage: **Bringen die
Frame-Haelfte der Aggregator-Tokens und beide Kamera-Tokens etwas, und wie
kodiert man sie am besten?**

## Die Dateien

| Datei | Inhalt |
|---|---|
| `GOAL.md` | Ziel, die zwei Vergleichsebenen, P1, die Latte, Wertungsmatrix, Ableseprotokoll |
| `RULES.md` | Zeit, eingefrorene Zone, freies Spielfeld, Compute, Wellenplan, PR |
| `LEDGER.md` | eine Zeile pro Versuch, gepflegt vom Orchestrator |
| `agents/<name>/NOTES.md` | Arbeitsnotizen, ein Ordner pro Agent |
| `runs/<jobid>-<slot>/` | kopierte `metrics.csv`, Logs, sbatch-Skripte |

`verify.sh` und `expected_curriculum.sha256` liegen eine Ebene hoeher und
werden **nicht** veraendert. `GOAL.md` und `RULES.md` im Elternordner gehoeren
zu Duell 2 und bleiben unangetastet.

## Der Kern in fuenf Zeilen

- **Latte:** Duell-2-Sieger C, paarweise pro Seed - 6060404 (s42), 6061173 (s43).
- **P1 (Pflicht):** `[cam_full(2048), mean(patches_full), max(patches_full)]`
  = 6144, mit Cs MLP-Encoder. Headline-Vergleich P1 gegen C, eine Variable.
- **Alle weiteren Arme** werden gegen P1 gerankt, aber gegen C auf die
  PR-Schwelle geprueft.
- **Eingefroren:** alle Trainings-Hyperparameter. **Frei:** Adapter *und*
  Encoder. Harte Bedingung: jeder Arm enthaelt die Frame-Haelfte.
- **PR ab Mittel +0.10**, Bestaetigung auf Seed 43 Pflicht.

## Vorgeschichte

- `../2026-07-27/` - Duell 1: der pooled Arm schlaegt hybrid und pointmap.
- `../2026-07-27-r2/` - Duell 2: kv200k + prefill 1024 + train_ratio 128 +
  act_entropy 0.1 ist der bestaetigte Sieger (Mittel +0.0546, unter der
  Schwelle, kein PR). Sieben Erkenntnisse und die offenen Faeden dort sind
  Pflichtlektuere - P3 stammt daraus.
