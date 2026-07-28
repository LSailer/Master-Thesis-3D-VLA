# Analyst-Notizen: Duell-1-Laeufe unter der Duell-2-Wertungsmatrix

Datum: 2026-07-27. Skript: `agents/analyst/score.py` (Stdlib, Pfad als Argument, wird fuer Welle-1 wiederverwendet).

Referenz (Score 0.0 per Definition): 6057641-aggpool-p2048 mit Treffer 1, softspl 0.0605, dtg 5.193, spl 0.0201, ms/Step 134.1, Episoden 18.

Formel: Score = 0.45*rel(Treffer,hoch) + 0.15*rel(softspl,hoch) + 0.15*rel(dtg,niedrig) + 0.10*rel(spl,hoch) + 0.10*rel(ms/Step,niedrig) + 0.05*rel(Episoden,hoch); Kappung Treffer [-1,+2], Rest [-1,+1]. "Letzter Wert" = Wert beim groessten step (CSV ist nicht step-sortiert). Nicht-numerische Zeilen (episode/goal) werden uebersprungen, zaehlen aber fuer N.

## Ablesewerte

| Lauf | Treffer | softspl | dtg | spl | ms/Step | Episoden | N |
|---|---|---|---|---|---|---|---|
| 6057639-hybrid-p2048 | 0 | 0.0544 | 5.327 | 0.0000 | 152.0 | 16 | 8001 |
| 6057641-aggpool-p2048 (REF) | 1 | 0.0605 | 5.193 | 0.0201 | 134.1 | 18 | 9001 |
| 6057871-aggpool-lottery | 1 | 0.0658 | 6.255 | 0.0034 | 71.4 | 40 | 19675 |
| 6057877-hybrid-lottery | 0 | 0.0894 | 4.938 | 0.0000 | 137.5 | 19 | 9503 |

## Einzelbeitraege (gewichtete, gekappte rel-Werte) und Gesamtscore

| Lauf | Treffer | softspl | dtg | spl | ms/Step | Episoden | Gesamtscore |
|---|---|---|---|---|---|---|---|
| 6057639-hybrid-p2048 | -0.4500 | -0.0150 | -0.0039 | -0.1000 | -0.0134 | -0.0056 | **-0.5878** |
| 6057641-aggpool-p2048 (REF) | +0.0000 | -0.0001 | -0.0000 | +0.0002 | +0.0000 | +0.0000 | **+0.0002** |
| 6057871-aggpool-lottery | +0.0000 | +0.0130 | -0.0307 | -0.0831 | +0.0467 | +0.0500 | **-0.0040** |
| 6057877-hybrid-lottery | -0.4500 | +0.0716 | +0.0074 | -0.1000 | -0.0026 | +0.0028 | **-0.4708** |

Selbsttest: Referenzlauf ergibt +0.0002 statt exakt 0.0 - Rundungsrest der auf 3-4 Stellen gerundeten Referenzwerte im Auftrag; Skript ok.

Kappungen: 6057871 Episoden roh +1.2222, gekappt auf +1.0000.

## Befunde

- **Wichtigster Punkt**: 6057871 (aggpool-lottery, kv200k, identisch zu Welle-1 Slot A) landet unter der neuen Matrix bei **-0.0040**, also praktisch punktgleich mit der Referenz. Der Geschwindigkeitsgewinn (ms/Step +0.0467) und die vielen Episoden (+0.0500, roh +1.22 gekappt) werden vom spl-Einbruch (0.0034 vs 0.0201, Beitrag -0.0831) und dtg (-0.0307) fast exakt aufgezehrt. Der Favorit gewinnt unter dieser Matrix NICHT ueber Tempo allein - er braucht mindestens einen zweiten Treffer (jeder unkappte Zusatztreffer = +0.45).
- Beide Hybrid-Laeufe verlieren fast ausschliesslich ueber Treffer=0 (-0.45) plus spl=0 (-0.10); alles andere ist Rauschen.
- Treffer dominiert die Matrix: Spanne des Treffer-Terms [-0.45, +0.90], alle anderen Terme zusammen maximal +/-0.55.
- Diagnostik (unscored): action/forward_pct liegt bei allen vier Laeufen bei 0.256-0.280, also nahe uniform (0.25) - keiner der Laeufe hat eine klare Vorwaertspolitik gelernt.

## Welle-1-Finallaeufe (folgt)

Zu scoren sobald sacct die Jobs als beendet meldet (~14:45 UTC): 6061173 (aggpool-b200k-tr128, Seed 43, Bestaetigung Fuehrender C), 6061174 (ent3em4), 6061175 (ent3em3), 6061176 (pointmap-p2048). Zusaetzlich abzulesen: action/forward_pct, episode/path_length.
