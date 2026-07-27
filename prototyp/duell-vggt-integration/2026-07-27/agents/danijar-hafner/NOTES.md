# danijar-hafner - Arbeitsnotizen

Alles, was dieser Agent probiert, findet und verwirft. Roh und fortlaufend.
Kein Politur-Anspruch, aber jede Zahl mit Quelle (SLURM-Job, Run-Dir, Datei:Zeile).

---

# 2026-07-27: Konfig-Hebel fuer das 30-Minuten-Budget

Vorbemerkung zur Ehrlichkeit: das Persona-Dossier unter
`/Users/lucamac/.../personas/Danijar Hafner/dossier.md` existiert auf diesem
Cluster-Dateisystem nicht. Ich arbeite hier aus dem gelesenen Code plus
meinen Prinzipien. Alles, was Extrapolation ist, ist unten als solche
markiert. Ich habe nie einen Agenten in Habitat trainiert und habe keine
dokumentierte Position dazu, was ein 3D-Prior einem World Model bringt.

Alle Pfade unten relativ zum Worktree
`/pfs/data6/home/ul/ul_student/ul_hfj15/Master-Thesis-3D-VLA/.claude/worktrees/dreamerv3-3d-features-deb5ea`.
Config-Dateien: `src/configs/agent_config.py`, `src/configs/trainer_config.py`
(nicht `src/r2dreamer/`, wie in der Aufgabenstellung angenommen).

## 0. Das Framing, bevor irgendein Knob gedreht wird

Rechnung mit den echten Defaults:

- `train_credit += acfg.train_ratio / batch_steps` mit
  `batch_steps = batch_size * seq_len` (`loops.py:427`, `loops.py:454`).
  512 / (16 * 64) = **0.5 Updates pro Env-Step**.
- 9000 Env-Steps minus 2048 Prefill = ~7000 Steps -> **~3500 Gradienten-
  Updates**. Davon liegen 1000 in der linearen LR-Warmup-Rampe
  (`agent_config.py:145`, verdrahtet in `agent.py:415-422`).
- Bei `lr = 4e-5` (`agent_config.py:141`) lernt in 3500 Updates weder das
  Reward-Head das `success_bonus`-Ereignis noch der Critic eine Wertfunktion,
  die 200 Schritte weit traegt.

Daraus folgt die einzige Aussage, die ich mit Ueberzeugung mache:

**In diesem Budget ist `metrics/sr` kein Lernergebnis, sondern eine Lotterie
ueber die Anzahl gewerteter Episoden.** Alle Episoden laufen ins Cap
(`src/environments/habitat.py:49`, `max_episode_steps = 500`; die Baseline
hat 59322 Steps / 120 Episoden = 494 Steps pro Episode). Bei ~7000
gewerteten Steps sind das **14 Episoden**. Mit der Random-SR von 3.84 Prozent
ist P(mindestens ein Erfolg) = 1 - 0.9616^14 = **42 Prozent**.

Die Zielfunktion ist also nicht "besser lernen", sondern:

1. Zahl der **gewerteten** Episoden maximieren (mehr Lose).
2. Die Policy nicht unter Random fallen lassen (Lose nicht entwerten).

Punkt 2 ist nicht hypothetisch: die CNN-Baseline liegt nach 59322 Steps bei
SR 0.02, also **unter** den 3.84 Prozent des Random-Agenten (LEDGER). Der
Gegner ist hier nicht "zu wenig gelernt", sondern "eine halbtrainierte Policy
ist schlechter als Wuerfeln".

Konsequenz, die unbequem ist und die ich trotzdem sage: ein Lauf, der so
gewonnen wird, sagt **nichts** ueber die Qualitaet der VGGT-Integration. Die
Varianz zwischen zwei Seeds derselben Config ist in diesem Budget groesser
als jeder Encoder-Effekt. Das GOAL.md gibt das unter "Ehrlicher Vorbehalt"
schon zu; ich verschaerfe es nur: es ist nicht "koennte im Rauschen liegen",
es **ist** Rauschen.

## 1. Was ueberhaupt ohne Code-Aenderung erreichbar ist

Der YAML-`args`-Block wird 1:1 zu CLI-Flags von `run.py` gerendert
(`scripts/slurm/launch.py:286`); die Flags sind abschliessend in
`src/r2dreamer/launch/parser.py` definiert. Erreichbar:

`prefill` (`parser.py:36`), `steps`, `act_entropy` (`parser.py:136`),
`lr` (`parser.py:224`), `train_ratio` (`parser.py:230`),
`batch_size` (`parser.py:212`), `seq_len` (`parser.py:218`),
`buffer_capacity`, `latent_preset`, `deter_size`, `stoch_*`, `decoder`,
`compute_dtype`, `val_*`, `log_every`, `actor_loss_weight`,
`value_loss_weight`, `repval_loss_weight`.

**Nicht** erreichbar, braucht je eine Code-Aenderung:
`imagination_horizon`, `horizon`, `lamb`, `unimix_ratio`, `warmup_steps`,
`kl_free`, `slow_target_fraction`, `agc_clip` (alle
`agent_config.py:140-172`) sowie `success_bonus`, `step_penalty`,
`max_episode_steps` (`habitat.py:49-52`). Das ist fuer die Reihenfolge
entscheidend: jeder Knob aus der zweiten Liste kostet einen Edit plus Smoke,
und ein Edit unter Zeitdruck ist die haeufigste Quelle fuer einen
gescheiterten Lauf.

## 2. Geordnete Liste

### [SOFORT 1] `act_entropy`: 3e-2 -> 1e-1

- **Wo**: `agent_config.py:166`, Flag `parser.py:136-142`.
- **Wirkung**: Der Actor-Verlust ist
  `-(logpi * stop_grad(adv) + cfg.act_entropy * entropy)`
  (`src/r2dreamer/behavior/loss.py:139-142`), der Advantage ist bereits durch
  `ret_scale` normalisiert (`loss.py:119-120`). Die Entropie konkurriert also
  gegen eine skalen-normierte Groesse; der Koeffizient ist direkt
  interpretierbar. Hoehere Entropie haelt die Policy nahe an der
  Gleichverteilung und damit nahe an der 3.84-Prozent-Chance, statt in die
  2 Prozent der Baseline zu kollabieren.
- **Warum genau hier**: Der Hilfetext im Repo sagt selbst, dass 3e-4 (der
  DreamerV3-Paper-Wert) "collapses the policy here" und dass 3e-2 der getunte
  Habitat-Wert ist (`parser.py:139-141`). Ihr wisst also bereits, dass dieser
  Knob in eurem Setup den Policy-Kollaps steuert. In einem 3500-Update-Lauf
  will man auf der sicheren Seite davon stehen. Nebenwirkung begrenzt, weil
  der Actor auf detachten imaginierten Features rechnet (`loss.py:122-127`),
  der Term also nicht in Encoder oder RSSM zurueckwirkt.
- **Risiko**: gering fuer den Lauf, hoch fuer die Interpretation. Wenn dieser
  Arm gewinnt, hat er wegen Exploration gewonnen, nicht wegen VGGT. So ins
  LEDGER eintragen. Ein zweiter Arm mit 3e-1 waere die reine
  Lotterie-Variante; die wuerde ich nicht als gewerteten Lauf abgeben.
- **Extrapolation**, kein gemessener Wert: 1e-1 ist ein 3.3-facher Schritt,
  bewusst klein genug, um nicht in ein anderes Regime zu springen.

### [SOFORT 2] `train_ratio`: 512 -> 256

- **Wo**: `agent_config.py:138` / `trainer_config.py:26`, Flag
  `parser.py:230`.
- **Wirkung**: halbiert die Updates pro Env-Step von 0.5 auf 0.25
  (`loops.py:454`) und kauft damit Wall-Clock fuer Env-Steps. Grobe
  Zerlegung der 219 ms/Step aus MEMORY (`house-points-pose-step-cost`):
  VGGT ~132 ms, `replay_sample` ~59 ms amortisiert, Rest ~28 ms
  Train-Compute amortisiert. Der trainierende Anteil ist ~87 ms von 219,
  also 40 Prozent. Halbieren spart ~43 ms/Step -> **~176 ms/Step, etwa 24
  Prozent mehr Env-Steps und damit 24 Prozent mehr Lose**.
- **Warum das die richtige Richtung ist**: Der Reflex bei knappem Budget ist,
  `train_ratio` hoch zu drehen, weil GPU-Updates billiger scheinen als
  Env-Steps. Das stimmt hier nicht, weil `replay_sample` mit ~118 ms pro
  Update-Aufruf (59 ms bei 0.5 Updates/Step) der zweitteuerste Posten
  ueberhaupt ist. Und selbst wenn es billig waere: 7000 statt 3500 Updates
  aendern nichts an einer Policy, die noch nicht in der Naehe von Lernen ist,
  waehrend 24 Prozent mehr Episoden direkt auf die Zielmetrik gehen.
- **Risiko**: Wenn die Annahme falsch ist und der Arm doch in ~3500 Updates
  etwas lernt, verschenkt man das. Dagegen spricht die Baseline, die mit
  59322 Steps und ~29660 Updates immer noch unter Random liegt.
- **Gegenprobe, falls Zeit ist**: derselbe Arm mit `train_ratio: 1024`. Wenn
  der schneller lernt, ist meine ganze Rahmung falsch, und das will man
  wissen. Billiges Falsifikationsexperiment.

### [SOFORT 3] `prefill`: 2048 -> 1024

- **Wo**: `parser.py:36`, aktuell 2048 in `duell_l3_hybrid_p2048.yaml` und
  `duell_l3_aggpool_p2048.yaml`.
- **Wirkung**: `prefill()` ruft `experience.step(action, summarize=False)`
  (`loops.py:377`). `summarize=False` heisst, dass fuer diese Episoden **kein
  `EpisodeSummary` gebaut wird** (`src/r2dreamer/experience.py:249`) und sie
  nie in `metrics/sr` landen. 2048 Prefill-Steps bei 500er-Cap sind also
  **vier ungewertete Lose**. Bei 14 Losen insgesamt fast 30 Prozent des
  Budgets, verworfen, obwohl der Random-Agent dort genau seine 3.84 Prozent
  Erfolgschance hat.
- **Warum nicht weiter runter**: Der Trainings-Gate ist
  `experience.buffer_size >= batch_steps` = 1024 (`loops.py:453`) und wird in
  jedem Step geprueft, oeffnet also von selbst. `prefill: 1024` heisst: Gate
  oeffnet exakt beim ersten Trainings-Step, kein Step wird verschenkt.
  `prefill: 0` waere theoretisch noch besser, aber `loops.py:353-361`
  beschreibt einen realen Bug (PERSIST_SCENE / `reset_for_scene` feuert
  sonst nicht, smoke 5738008). Diesen Pfad wuerde ich unter Zeitdruck nicht
  anfassen.
- **Risiko**: praktisch keins. Reine Buchhaltung.

### 4. `seq_len` runter: nicht ohne gleichzeitige `train_ratio`-Korrektur

Die naive Version ist eine Falle. `train_credit += train_ratio / (batch_size
* seq_len)` (`loops.py:454`): `seq_len: 32` bei unveraendertem `train_ratio:
512` ergibt **1.0 Update pro Env-Step statt 0.5**, also doppelt so viele
(halb so teure) Updates plus doppelt so viele `replay_sample`-Aufrufe und
doppelten Dispatch-Overhead. Netto wird der Lauf eher **langsamer**. In
diesem Repo ist `train_ratio` eine Transitions-Rate, keine Update-Rate. Wer
`seq_len` halbiert und die Update-Rate halten will, muss `train_ratio`
mithalbieren. Erwartete Wirkung dann: etwas billigeres Sampling bei gleicher
Update-Zahl, aber kleiner als Hebel 2, und es aendert zusaetzlich das
RSSM-Kontextfenster. Nicht sofort.

### 5. `imagination_horizon` / `horizon` / `lamb`

- Werte 15 / 333 / 0.95 (`agent_config.py:162-164`), Discount
  `1 - 1/333 = 0.997` (`loss.py:102`).
- Argument fuer `horizon` ~50 (Discount 0.98): in 3500 Updates ist die
  einzige lernbare Groesse das dichte `geodesic_delta`
  (`habitat.py:513-518`). Ein Discount von 0.997 bittet den Critic, ueber 333
  Schritte zu integrieren, was er nicht kann; ein kuerzerer Horizont macht
  das Ziel leichter fittbar.
- **Warum trotzdem nicht sofort**: kein CLI-Flag, also Code-Edit, und die
  Wirkung geht auf Lernqualitaet statt auf die Zahl der Lose. In 30 Minuten
  zahlt sich das nicht aus. Fuer die 2M-Laeufe dagegen eine echte Frage:
  500-Schritt-Episoden gegen einen 333er-Horizont ist eine ungewoehnliche
  Kombination.
- Kennzeichnung: Extrapolation aus Prinzipien, ich habe dazu keine gemessene
  Position.

### 6. Reward-Shaping (`success_bonus`, `step_penalty`)

Erlaubt, aber in diesem Budget **wirkungslos**, deshalb weit unten.

- `success_bonus 10.0` (`habitat.py:52`, angewandt `habitat.py:516-517`) kann
  nur wirken, wenn das Reward-Head das Ereignis mehrfach in einem Batch
  sieht. Bei null bis einem Erfolg im gesamten Replay ist der
  Gradientenbeitrag praktisch null. Auf 100 gesetzt aendert das nichts,
  erzeugt aber einen massiven Ausreisser in der Twohot-Verteilung
  (`twohot_bins: 255`, `agent_config.py:129`) und riskiert, die
  Reward-Vorhersage fuer die dichten Deltas zu verschlechtern. Nettoverlust.
- `step_penalty -0.01` ist eine Konstante pro Schritt. Eine additive
  Konstante verschiebt Returns, aber der Advantage in `loss.py:120` ist eine
  Differenz und danach durch `ret_scale` normiert. Effekt auf die Policy
  zweiter Ordnung. Nicht anfassen.
- Was theoretisch wirken wuerde, ist ein dichteres Erfolgssignal (abgestufter
  Bonus ab 1.0 m statt einer Stufe bei 0.2 m). Das ist aber keine
  Reward-Aenderung mehr, sondern eine Aenderung des Problems, siehe 7.

### 7. Die zwei Knoepfe, die ich bewusst NICHT empfehle

Beide haben die hoechste erwartete Wirkung auf `metrics/sr` von allem hier,
und genau deshalb stehen sie unten.

- **`GOAL_RADIUS = 0.2`** (`habitat.py:36`). Sehr enger Radius; in der
  Habitat-ObjectNav-Literatur ist 1.0 m ueblich (hier bin ich ausserhalb
  meines Gebiets, das gehoert der Embodied-AI-Seite, aber der Zahlenwert
  steht im Code). Erhoehen wuerde die Erfolgsrate vervielfachen.
- **`max_episode_steps = 500`** (`habitat.py:49`). Auf 150 gesenkt bekaeme
  man statt 14 Episoden ueber 45, also dreimal so viele Lose.

Beide aendern nicht die Policy, sondern **die Definition der Zielmetrik**.
Die Baseline 6056750 lief mit 0.2 m und 500 Steps; ein Arm mit anderen
Werten ist bei "SR bei Step N" nicht mehr vergleichbar, und das Duell misst
dann nur noch, wer die Metrik weiter aufgeweicht hat. Das ist gameable, und
ein gameable Benchmark ist in dem Moment wertlos, in dem jemand ihn gamed.
Wenn ihr das ausprobieren wollt, dann als **Diagnostik-Lauf ausserhalb der
Wertung**, im LEDGER als solcher markiert. Die eine Frage, die so ein Lauf
ehrlich beantwortet und die wirklich interessant ist: *wie viele der
Random-Erfolge passieren in den ersten 150 Schritten?* Wenn es die meisten
sind, ist das 500er-Cap fuer alle Arme Zeitverschwendung, und dann sollte man
das Curriculum aendern, offen und fuer alle Arme gleich.

### 8. Kleinkram aus dem Code-Lesen

- `val_every: 0` ist Default (`trainer_config.py:53`). Unbedingt so lassen:
  ein Val-Loop verbrennt Wall-Clock in Episoden, die nicht in `metrics/sr`
  zaehlen.
- `decoder: False` ist Default (`agent_config.py:115`). So lassen. Bei
  `decoder=True` haengt an jedem Log-Schritt zusaetzlich
  `agent.reconstruct(batch)` plus `device_get` (`loops.py:471-479`).
- `checkpoint_every: 50_000` (`trainer_config.py:29`): bei ~9000 Steps kein
  Checkpoint, und weil der Lauf per TIMEOUT endet, auch kein finaler. Fuer
  die Wertung egal, `metrics.csv` genuegt. Nur nicht ueberrascht sein.
- `warmup_steps: 1000` (`agent_config.py:145`, `agent.py:415-421`): bei ~3500
  Updates liegt fast ein Drittel des Laufs in der LR-Rampe, bei
  `train_ratio: 256` mehr als die Haelfte. Kein CLI-Flag. Fuer die
  SR-Lotterie irrelevant, aber der Grund, warum man aus diesen Loss-Kurven
  **nichts** ueber Konvergenz ablesen darf.
- `log_every: 250` mit dem gelatchten Log-Pfad (`loops.py:455-470`): bei 9000
  Steps rund 36 Log-Zeilen. Fuer die Auswertung eng. `50` waere billiger als
  es klingt; Kosten sind ein `materialize=True` pro Log (`loops.py:465`).

## 9. Was ich stattdessen fragen wuerde

Die produktivste Frage ist nicht, welcher Knob den 30-Minuten-Lauf gewinnt,
sondern: **welches ist die billigste Umgebung, die dieselbe Herausforderung
reproduziert?** Wenn das Ziel ist, Integrationsvarianten fuer VGGT-Features
zu vergleichen, dann ist eine Umgebung, in der die Zielmetrik ein Muenzwurf
ueber 14 Episoden ist, das falsche Messgeraet, egal wie man sie
konfiguriert. `src/environments/crafter.py` liegt im selben Repo. Eine
Integrationsvariante, die dort nicht wirkt, wird auf HM3D auch nicht wirken,
und dort dauert eine Antwort Minuten statt Tage.

Zweitens, und das haette ich zuerst gefragt: **was gibt das 3D-Feature dem
World Model, das Pixel nicht geben?** Es gibt eine messbare Version dieser
Frage, die kein SR-Budget braucht: fester, eingefrorener Replay-Batch, dann
messen, wie gut der Dynamik-Prior den naechsten Posterior vorhersagt
(`scale_dyn` / `scale_rep`, `agent_config.py:153-154`), CNN gegen Hybrid, bei
gleicher Parameterzahl. Laeuft in Minuten, ist deterministisch, und
beantwortet die These aus GOAL.md direkt statt ueber eine Erfolgs-Lotterie.
Der Repo-Pfad existiert bereits: `overfit_one_batch`
(`trainer_config.py:68`).

Und die Warnung, die ich fuer die wertvollste halte: die zwei Kritiken, die
bei "extra Modalitaet in ein DreamerV3-World-Model" garantiert kommen, sind
"ist das nicht einfach DreamerV3 mit einem zusaetzlichen Input?" und "sind es
die 3D-Features oder die zusaetzliche Kapazitaet?". Dynalang (ICLR 2024) hat
die zweite mit Varianten davon beantwortet, *wie* die Modalitaet eintritt,
statt mit einer parameter-gematchten Kontrolle, und wurde trotz Scores
6/6/5 abgelehnt. Ein parameter-gematchter CNN-Arm ist billiger zu bauen als
jede Encoder-Variante in diesem Repo und deckt genau diese Luecke.
