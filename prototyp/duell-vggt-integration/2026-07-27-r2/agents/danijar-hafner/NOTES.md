# Danijar Hafner - Notizen Duell 2 (2026-07-27-r2)

Modus: brainstorm. Zustaendig fuer World Model / Actor / Objective, nicht fuer
VGGT-Feature-Qualitaet (das ist Jianyuans Seite).

Vorbemerkung zur Quellenlage: alle Aussagen hier stammen entweder aus dem Code
dieses Repos (mit `datei.py:zeile`), aus den `metrics.csv` der Runde 1, oder sind
als **Inferenz** markiert. DreamerV3-Referenzwerte zitiere ich aus dem
Gedaechtnis an unsere Config, nicht aus einem verifizierten Dokument - bitte
gegenpruefen, bevor eine solche Zahl in die Arbeit wandert.

---

## 0. Der Befund, der alles andere ueberschreibt

Ich habe zuerst instrumentiert statt geraten. In allen vier Laeufen der Runde 1
steht die Aktionsverteilung der letzten Episode bei:

| Run | fwd | stop | Episoden | softspl | dtg | Treffer |
|---|---|---|---|---|---|---|
| 6057639 hybrid-p2048 | 0.280 | 0.166 | 16 | 0.0544 | 5.327 | 0 |
| 6057641 aggpool-p2048 (Referenz) | 0.256 | 0.208 | 18 | 0.0605 | 5.193 | 1 |
| 6057871 aggpool-lottery (kv200k) | 0.256 | 0.244 | 40 | 0.0658 | 6.255 | 1 |
| 6057877 hybrid-lottery | 0.258 | 0.314 | 19 | 0.0894 | 4.938 | 0 |

Quelle: `prototyp/duell-vggt-integration/2026-07-27/runs/<jobid>/metrics.csv`,
letzter geloggter Wert von `action/{forward,stop}_pct`, `metrics/softspl`,
`metrics/dtg`, `episode/count`; Treffer = Zeilen `episode/success == 1`.

**0.25 pro Aktion bei 4 Aktionen ist die Uniformverteilung.** Kein Arm hat in
30 Minuten irgendeine Aktionspraeferenz gelernt. Die Policy ist ein Random Walk,
und zwar in jedem Arm derselbe. Damit ist die Frage "welche 3D-Integration lernt
schneller an" in diesem Fenster nicht gestellt worden: es lernt keine.

Zweiter Befund aus derselben Zeile, Referenzlauf: `episode/steps 500`,
`action/forward_pct 0.256` -> ca. 128 Forward-Aktionen, nominal 0.25 m each =
32 m intendierte Strecke. Tatsaechlich `episode/path_length 5.519`. Also sind
ungefaehr 80 % der Forward-Schritte gegen Geometrie gelaufen. Der Agent klebt an
Waenden. `collision_rate` wird nicht geloggt (`track_collision_rate` offenbar
aus), aber die Arithmetik ist eindeutig.

**Konsequenz fuer die Frage "was verbessert die Annaeherung":** der bindende
Constraint ist nicht die Repraesentation und nicht das World Model. Es ist, dass
der Actor-Gradient das Entropie-Regularisierungsglied nicht schlaegt. Alles
andere in dieser Notiz ist der Versuch, das zu beheben.

---

## 1. Warum die Policy uniform bleibt - zwei Faktoren, die sich multiplizieren

### 1a. `act_entropy` ist um zwei bis drei Groessenordnungen zu hoch

- Repo-Default: `act_entropy: float = 3e-2` (`src/configs/agent_config.py:166`).
- Duell-Arme: `act_entropy: 0.1`
  (`scripts/slurm/configs/duell2_l3_aggpool_b200k_tr128.yaml`).
- DreamerV3-Referenzwert, den ich aus unserer Config erinnere: **3e-4**.
  (Als erinnert markiert, nicht verifiziert.)

Das heisst: der Repo-Default liegt bei ca. 100x der Referenz, die Duell-Arme bei
ca. 333x. Der Actor-Loss ist
`-(logpi * adv + cfg.act_entropy * entropy)`
(`src/r2dreamer/behavior/loss.py:141`). Bei 4 Aktionen ist die Uniform-Entropie
ln 4 = 1.386. Wenn `adv` klein ist, optimiert dieser Loss nichts anderes als
"bleib uniform". Genau das messen wir.

Das ist bei euch kein Bug, sondern eine bewusst gesetzte Zahl - aber es ist die
Zahl, die in Runde 1 als "Lotterie-Knopf" hochgedreht wurde, um mehr
Zufallstreffer zu erzeugen. In einer Matrix, die zu 0.30 aus dichten
Annaeherungsmetriken besteht, arbeitet dieser Knopf jetzt gegen euch. Wer `dtg`
runter und `softspl` rauf will, muss ihn **runter**drehen, nicht rauf.

### 1b. Der Return-Scale-Clamp macht `adv` zusaetzlich klein

`src/r2dreamer/behavior/return_ema.py:35`:

    scale = jnp.maximum(state[1] - state[0], 1.0)

und in `src/r2dreamer/behavior/loss.py`: `adv = (ret - imag_value[:, :-1]) / ret_scale`.

Das `max(..., 1.0)` ist korrekt und steht so auch in DreamerV3 - es soll
verhindern, dass in einem fast reward-freien Regime das Rauschen verstaerkt wird.
Der Preis: liegt die Perzentil-Spanne der Returns unter 1, wird `adv` **nicht**
auf Einheitsskala gebracht, sondern bleibt klein.

Wie klein ist sie hier? `episode/reward -6.56` bei 500 Steps, `metrics/reward
-4.32` als Fenstermittel (Referenzlauf). Bei `step_penalty -0.01` sind das allein
-5.00 pro Episode aus der Konstante; der geodesic_delta-Anteil summiert sich also
auf ungefaehr -1.5 bis +0.7 pro **ganzer Episode**. Pro Step liegen die Deltas
damit im Bereich 0.0x m. Ein 15-Schritt-Lambda-Return (`imagination_horizon: 15`,
`agent_config.py:162`) hat entsprechend eine Spanne deutlich unter 1. **Der Clamp
ist also mit hoher Wahrscheinlichkeit aktiv, und `adv` ist systematisch
untergewichtet.** (Inferenz aus den Reward-Zahlen; nicht gemessen, weil
`ret_scale` nirgends geloggt wird - `agent.py:737-739` schreibt nur `opt_loss`,
`total_loss`, `nan_skipped`.)

**1a und 1b multiplizieren sich.** Ein zu grosser Entropie-Term und ein zu klein
skalierter Advantage sind derselbe Fehler von zwei Seiten. Beide Hebel greifen am
gleichen Verhaeltnis an, und beide sind billig.

### 1c. Der `success_bonus 10.0` ist im 1-Treffer-Regime aktiv schaedlich

`src/configs/agent_config.py:172`. Sobald ein einziger Erfolg im Replay landet,
springt die Return-Spanne von <1 auf ~10. Die EMA hat `alpha=0.01`
(`return_ema.py:15`), also eine Zeitkonstante von ~100 Updates. In diesen ~100
Updates werden alle dichten geodesic_delta-Signale durch ~10 geteilt und
verschwinden. Ein Treffer loescht also fuer 100 Gradientenschritte genau das
Signal, das `dtg` und `softspl` bewegen soll. Bei 0-1 Treffern pro Slot ist das
kein grosser Effekt, aber die Richtung ist eindeutig falsch.

---

## 2. Antworten auf die vier Fragen

### Frage 1: Reward-Shaping

`geodesic_delta` **ist** bereits potential-based shaping mit Phi(s) = -d(s) und
gamma = 1. Das ist die richtige Form, sie ist policy-invariant, und ich wuerde
sie nicht ersetzen. Ihr habt nicht ein Shaping-Problem, ihr habt ein
**Skalierungs**problem.

Was hilft, in dieser Reihenfolge:

1. **Reward-Skalierung x10.** Multipliziert man `geodesic_delta` mit ~10, liegt
   die Return-Spanne ueber dem Clamp und `adv` bekommt eine echte Einheitsskala.
   Das ist mathematisch fast dasselbe wie `act_entropy` durch 10 zu teilen -
   deshalb wuerde ich erst den Entropie-Knopf drehen (existiert als CLI-Flag) und
   die Reward-Skalierung nur, wenn das nicht reicht.
2. **`step_penalty` ist wahrscheinlich irrelevant fuer den Policy-Gradienten.**
   Eine Konstante pro Step in einer Episode fester Laenge ist ein konstanter
   Offset auf den Return, und der faellt in `adv = ret - value` heraus, sobald der
   Critic ihn gelernt hat. Er kostet aber Dynamikbereich im Reward-Head
   (`twohot_bins: 255`) und laesst `metrics/reward` bei -4.3 stehen, was jede
   Reward-Kurve unlesbar macht. Auf 0 setzen ist sauber und harmlos.
   (Inferenz, nicht gemessen.)
3. **`success_bonus` von 10.0 auf ~1.0.** Begruendung in 1c.
4. **Distanz-normalisiertes Delta** (`delta / start_geodesic`) wuerde Episoden
   verschiedener Laenge auf dieselbe Return-Skala bringen. Konzeptionell die
   sauberste Variante, aber sie aendert die Reward-Semantik und braucht einen
   Wrapper. **Nicht im 30-Minuten-Fenster.** Ein Skalar tut fast dasselbe.

**Zum Reward-Wrapper ausserhalb von `src/environments/`: die Muehe ist es nicht
wert.** Ein Wrapper muesste den ganzen `ObservationFrame` durchreichen
(`image, is_first, previous_action, reward, done, softspl, dtg, spl, success,
scene_id, episode_id, step`) - und `softspl`, `dtg`, `spl`, `success` sind genau
die gewerteten Groessen. Ein Fehler im Wrapper korrumpiert also still die
Wertung. Der Reward betritt den Agenten an genau **einer** Stelle:

    src/r2dreamer/experience.py:257    reward=float(frame.reward),

Ein `reward_scale`-Feld in der Config und ein `* cfg.reward_scale` an dieser
Zeile ist derselbe Effekt, drei Zeilen Code, und kann die gewerteten Metriken
strukturell nicht anfassen. Das ist die Version, die ich bauen wuerde.

### Frage 2: Actor / Exploration

- **`act_entropy`: runter, nicht rauf.** Zwei Arme: `3e-4` (Referenz) und `3e-3`.
  Beide via existierendem `--act_entropy`. Kein Code.
  Risiko ehrlich benannt: bei niedriger Entropie kann die Policy in "nur drehen"
  kollabieren; dann bleibt `dtg` auf der Startdistanz stehen und `softspl` bei 0.
  Das erkennt man sofort an `action/forward_pct`. Diagnose vor Interpretation:
  **erst `action/forward_pct` lesen, dann ueber dtg reden.**
  `unimix_ratio: 0.01` (`agent_config.py:167`) bleibt als Exploration-Boden.
- **`imagination_horizon`: nicht anfassen** (kein Flag, und 15 ist bei
  Ein-Schritt-Credit fuer "geh vorwaerts" ausreichend). Kuerzer waere ein reiner
  Compute-Hebel, aber `train_ratio` ist derselbe Hebel mit einem Flag.
- **`discount`: nicht anfassen.** `horizon: 333` -> disc 0.997
  (`behavior/loss.py`, `agent_config.py:163`). Das Ziel liegt >100 Schritte
  entfernt; ein kuerzerer Discount wuerde die Annaeherung aktiv verlernen.
  `lamb: 0.95` ebenso in Ruhe lassen.

### Frage 3: Replay

- **`buffer_capacity`: nicht anfassen.** 500_000 (`agent_config.py:175`) bei
  10-25k gesammelten Steps. Es wird in diesem Fenster nie etwas evictet. Der
  Knopf ist in diesem Regime ein No-op.
- **`batch_size 32` + `seq_len 32` statt `16` + `64`** ist der einzige
  Replay-Hebel, den ich fuer plausibel halte. `loops.py:427`:
  `batch_steps = batch_size * seq_len` bleibt bei 1024, also bleiben die
  Update-Kadenz (`train_credit += train_ratio / batch_steps`, `loops.py:454`) und
  die Prefill-Paritaet exakt gleich - kein Risiko fuer den Logging-Bug. Gewonnen
  wird: doppelt so viele unabhaengige Startzustaende fuer die Imagination (`B*T`
  Rollout-Starts in `behavior/loss.py`), also ein diverserer Actor-Gradient bei
  gleichen Kosten, und kuerzeres BPTT, das fruehe Dynamik schneller anlernt.
  Risiko: 32 Steps Kontext statt 64. Bei 500-Step-Episoden sind 32 Steps immer
  noch ~8 m Weg, das reicht fuer lokale Navigation.
  Beide Flags existieren: `--batch_size`, `--seq_len`.

### Frage 4: Was in <30 min machbar ist, und was ich NICHT tun wuerde

**Machbar ohne jede Codezeile (nur YAML `args:`):**

| Aenderung | Flag | Erwartung |
|---|---|---|
| `act_entropy: 3e-4` | existiert | Policy verlaesst Uniform; Richtung von dtg offen |
| `act_entropy: 3e-3` | existiert | konservative Version davon |
| `batch_size: 32, seq_len: 32` | existiert | diverserer Actor-Gradient, gleiche Kadenz |

**Machbar in ~10-15 min Code:** `reward_scale`-Feld in
`src/configs/agent_config.py` + `--reward_scale`-Flag in
`src/r2dreamer/launch/parser.py` + Multiplikation an `experience.py:257`. Dazu
`--success_bonus`, damit 10.0 -> 1.0 ohne YAML-Trick geht (das Feld existiert in
`agent_config.py:172` und `src/shared/configs.py:16`, hat aber kein Flag; gesetzt
wird es bei `src/main.py:242`).

**Was ich NICHT tun wuerde:**

1. **Episoden per Wrapper frueher abschneiden, um `Episoden` zu farmen.** 0.05
   Gewicht, und es macht `dtg`/`softspl` gegen die Referenz unvergleichbar, weil
   beide Episodenend-Groessen sind und von der Episodenlaenge abhaengen
   (`habitat.py:477-489`). Das Ledger von Runde 1 hat den Step-Cap-Trick schon als
   "riecht nach Metrik-Gaming" markiert. Ich halte das fuer richtig.
2. **`buffer_capacity` variieren.** No-op, siehe oben.
3. **`discount` / `horizon` / `lamb` variieren.**
4. **Einen Reward-Wrapper als neue Datei bauen.** Begruendung in Frage 1.
5. **`prefill 0`.** PERSIST_SCENE-Bug, `loops.py:353-361`, dokumentiert.
6. **Semantisches Shaping ("Ziel im Bild") oder eine Ziel-Detektion.** Braucht
   einen Detektor, ist ein anderes Projekt.
7. **Am `max(..., 1.0)`-Clamp selbst drehen.** Der ist absichtlich da. Skaliert
   den Reward, nicht den Clamp.

---

## 3. Die unbequeme Ehrlichkeit zur Wertungsmatrix

Rechnet man die Matrix aus GOAL.md nach:

- Ein zusaetzlicher Treffer (1 -> 2) = rel +1.00 = **+0.45 Punkte**.
- `softspl` verdoppeln (0.0605 -> 0.121) = gekappt bei +100 % = **+0.15 Punkte**.
- `dtg` halbieren (5.193 -> 2.60) = rel +0.50 = **+0.075 Punkte**.

**Ein einziger Zufallstreffer ist dreimal so viel wert wie eine Verdopplung von
softspl.** Die Matrix ist also weiterhin lotteriedominiert; die dichten Metriken
sind ein Trostpreis. Dazu kommt: die vier Runde-1-Laeufe streuen bei praktisch
identischer (uniformer) Policy zwischen softspl 0.054 und 0.089 und dtg 4.94 und
6.25. Diese Streuung ist **nicht** Integrationsqualitaet, sie ist die
Haeuser-/Startdistanz-Ziehung von 16-40 Episoden. `episode/shortest_path` der
letzten Episode im Referenzlauf ist 2.40 m, `metrics/dtg` 5.19 - die
Startdistanzen variieren also um mehr als der gemessene Effekt.

Was daraus folgt: **verlaesslich bankbar sind nur `ms/Step` (0.10) und `Episoden`
(0.05).** kv200k liefert davon fast alles: 71.5 vs 134.1 ms/Step = rel +0.467 ->
+0.047, und 40 vs 18 Episoden = gekappt +1.00 -> +0.050, zusammen ~+0.097.
Nachgerechnet ergibt 6057871 (kv200k) unter der neuen Matrix trotzdem nur einen
Score von ungefaehr **-0.00**, weil `spl` (0.0034 vs 0.0201, rel -0.83 ->
-0.083) und `dtg` (6.25, rel -0.20 -> -0.031) den Speed-Gewinn wieder
auffressen. Das ist der Kernbefund fuer die Planung: **Speed allein reicht knapp
nicht, wenn dabei spl und dtg schlechter ziehen.** Ein Arm braucht Speed **und**
eine Policy, die nicht mehr uniform ist.

Und noch etwas, das in die Arbeit gehoert, nicht in die Wertung: dass alle vier
Arme bei `forward_pct` 0.256-0.280 stehen, heisst, dass dieses
30-Minuten-Protokoll die eigentliche Forschungsfrage - hilft ein 3D-Prior einem
modellbasierten Agenten - nicht beantworten kann. Es misst Durchsatz und
Ziehungsglueck. Als Iterationswerkzeug ist das voellig in Ordnung; als Evidenz
fuer die These ist es nichts. Diesen Satz wuerde ich lieber selbst schreiben, als
ihn spaeter von einem Reviewer zu hoeren.

---

## 4. Vorschlag fuer die naechste Welle (4 Slots)

Alle vier nehmen den kv200k-Adapter, weil Speed der einzige bankbare Hebel ist.

| Slot | Aenderung ggue. kv200k-Basis | Zweck |
|---|---|---|
| A | `act_entropy: 3e-4` | verlaesst die Policy die Uniformverteilung? |
| B | `act_entropy: 3e-3` | konservative Variante von A |
| C | `act_entropy: 3e-4` + `batch_size 32` + `seq_len 32` | A plus diverserer Actor-Gradient |
| D | `act_entropy: 3e-4` + `reward_scale 10` + `success_bonus 1` (braucht die 3 Codezeilen) | Advantage ueber den Clamp heben |

**Ablesereihenfolge, bevor irgendein Score interpretiert wird:**
`action/forward_pct` -> `episode/path_ratio` -> `metrics/dtg` -> Treffer.
Wenn `forward_pct` bei 0.25 bleibt, hat der Arm nichts gelernt und der Score ist
eine Ziehung, egal wie er aussieht. Wenn `forward_pct` steigt und `path_ratio`
faellt, bewegt sich der Agent zielgerichteter - das ist das erste echte
Lernsignal, das dieses Setup produzieren kann, und es ist billiger und
aussagekraeftiger als jede Erfolgsrate bei n=18.

Als Diagnose ausserdem sinnvoll und billig: `ret_scale` aus
`return_ema.get_stats` in die Metriken loggen (`agent.py:737-739` ist die
Stelle). Erst instrumentieren, dann Schwellen setzen - ich haette lieber die
gemessene Return-Spanne als meine Inferenz aus Abschnitt 1b.

---

## 5. Was ausserhalb meines Bereichs liegt

Ob Aggregator-Tokens, Pointmaps oder World-Points die bessere Geometrie tragen,
kann ich nicht beurteilen - dazu habe ich nichts publiziert und keine Meinung,
die etwas wert waere. Das ist Jianyuans Notiz.

Meine Frage an diese Feature-Varianten ist eine andere, und sie ist die einzige,
die ich hier stellen kann: **welches Lernsignal betritt durch den 3D-Prior das
World Model, das durch Pixel nicht hereinkaeme, und an welcher Metrik wuerde man
es sehen?** Solange `action/forward_pct` in allen Armen bei 0.25 steht, gibt es
auf diese Frage keine Antwort aus den Daten - der Actor wird gar nicht getrieben,
also kann keine Repraesentation sich beweisen. Deshalb halte ich es fuer richtig,
in dieser Runde zuerst den Actor zum Laufen zu bringen und danach wieder
Feature-Varianten zu vergleichen. Andernfalls vergleicht man in beiden Armen
denselben Random Walk.
