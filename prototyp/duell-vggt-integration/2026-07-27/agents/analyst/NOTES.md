# analyst - Arbeitsnotizen

Alles, was dieser Agent probiert, findet und verwirft. Roh und fortlaufend.
Kein Politur-Anspruch, aber jede Zahl mit Quelle (SLURM-Job, Run-Dir, Datei:Zeile).

## Ableseprotokoll

Umgesetzt als `prototyp/duell-vggt-integration/2026-07-27/readout.sh`, damit
jede Zahl reproduzierbar aus derselben Quelle kommt und nicht aus einem
handgetippten `grep`. Das Skript folgt `PLAN.md:58-68`:

1. `N` = groesster Step in der eigenen `metrics.csv`.
2. Eigene SR = letzter `metrics/sr`-Wert.
3. Baseline-SR = `metrics/sr` der Baseline beim groessten geloggten Step `<= N`.
4. Sekundaer: `episode/count`, `episode/steps`, `metrics/dtg`,
   Aktionsverteilung, `perf/ms_per_step_interval` und die gerechneten
   `1800 s / N * 1000` ms/Step aus `GOAL.md:55`.

Zusaetzlich zaehlt es die Erfolge direkt (`episode/success == 1`), weil
`metrics/sr` bei weniger als 100 Episoden ein Quotient mit kleinem Nenner ist
und ein einzelner Erfolg die Zahl stark bewegt. Die rohe Erfolgszahl ist die
ehrlichere Groesse und gehoert in jede Auswertung daneben.

## Fallstrick, der die Interpretation bestimmt

`metrics/sr` ist ein gleitender Mittelwert ueber bis zu 100 Episoden
(`RULES.md:24-26`). Ein 30-Minuten-3D-Lauf sieht aber nur ~16 Episoden. Der
Nenner ist also die Episodenzahl, nicht 100. Folge: **ein einziger Erfolg
erzeugt eine hohe SR und die Zahl faellt danach monoton**, weil jede weitere
erfolglose Episode den Nenner erhoeht, ohne den Zaehler zu bewegen.

Genau dieses Muster steht auch in der Baseline: ein Erfolg bei Step 14042
(1/29 = 0.0345), danach 1/30, 1/31, ... bis 1/42. Wer nur den Endwert liest,
sieht eine "faellende Performance", wo in Wahrheit nur der Nenner waechst.

Konsequenz fuer den Vergleich: SR-Werte aus Laeufen mit stark
unterschiedlicher Episodenzahl sind nicht direkt kommensurabel. Deshalb wird
neben der SR immer `episode/count` und die absolute Erfolgszahl notiert.

## Zwischenstand Welle 1 (waehrend die Jobs noch liefen)

Abgelesen um Cluster-Zeit 10:35, also vor Ende der 30 Minuten. Nur als
Verlaufsbeleg, nicht als Ergebnis.

| | 6057316 aggregator-pooled | 6057317 pointmap-pose |
|---|---|---|
| Step | 3501 | 3301 |
| Episoden | 7 | 6 |
| Erfolge | **1** | 0 |
| `metrics/sr` | 0.1429 | 0.0 |
| `metrics/spl` | 0.0518 | 0.0 |
| `metrics/dtg` | 4.82 | 4.35 |
| `perf/ms_per_step_interval` | 122.3 | 143.6 |
| Baseline-SR beim naechstliegenden Step | 0.0 (Step 3499) | 0.0 (Step 2999) |

Der Erfolg von 6057316 faellt in Episode 5, geloggt bei Step 2404 mit
`metrics/sr` 0.2 (= 1/5), danach 1/6 = 0.1667 und 1/7 = 0.1429.

Bemerkenswert, aber mit n = 1 nicht mehr als eine Notiz wert: der Erfolg liegt
bei **Seed 42 und demselben Curriculum** in den ersten Episoden, in denen die
CNN-Baseline mit identischem Seed 28 Episoden lang keinen einzigen Erfolg
hatte (erster Baseline-Erfolg bei Step 14042).

## Was in der metrics.csv am HEAD ankommt

`PROBLEMS.md:47-79` sagt, Trainingsmetriken wuerden nie geloggt. In den
Duell-Laeufen mit `log_every: 100` sind sie **vollstaendig da**:
`total_loss`, `loss/{dyn,rep,rew,value,policy,con,barlow,repval}`,
`latent/{prior,posterior}_entropy`, `params/encoder_l2`, `opt_loss`,
`nan_skipped` und die `perf/*`-Keys inklusive `perf/ms_per_step_interval`.

Zwei mogliche Erklaerungen, nicht auseinandergehalten und daher offen:
`log_every: 250 -> 100` verschiebt die Paritaet, die den in `PROBLEMS.md`
beschriebenen Bug ausloest; oder der Fix `67ed1c1 fix(loops): latch the
pending train-metric log until an update runs` (in `main` ueber Merge
`ccedf14`) hat das Problem ohnehin schon behoben. Wer das trennen will,
braucht einen Lauf am HEAD mit `log_every: 250`. Fuer das Duell irrelevant,
fuer `PROBLEMS.md` ein Update wert.
