# launcher - Arbeitsnotizen

Alles, was dieser Agent probiert, findet und verwirft. Roh und fortlaufend.
Kein Politur-Anspruch, aber jede Zahl mit Quelle (SLURM-Job, Run-Dir, Datei:Zeile).

## Umgebung

Der Agent sitzt auf dem Mac, nicht auf dem Cluster. Alles laeuft ueber
`ssh uc3` (Login-Node uc3n991). Ein bestehender ControlMaster-Socket erzeugt
bei jedem Aufruf zwei Rauschzeilen (`mux_client_request_session`,
`ControlSocket ... already exists`) - harmlos, aber jede Ausgabe muss danach
gefiltert werden, sonst landet der Muell in den Notizen.

## Sackgasse: Worktree unter /tmp (kostete zwei Jobs, ~12 Minuten)

**Symptom.** Zwei Jobs hintereinander (6057269, 6057297) gingen sofort in
RUNNING auf `uc3n082` und waren nach 21-23 Sekunden weg. `sacct` meldet beide
Male `FAILED`, ExitCode `2:0`. **Kein Logfile, kein Output-Verzeichnis, kein
Traceback** - nicht einmal die erste `echo`-Zeile des Wrapper-Skripts.

**Erste, falsche Diagnose.** Verdacht war `launch.py:227-228`: `#SBATCH
--output={log_dir}/slurm-%j.out` mit `log_dir = config.output_dir`
(`launch.py:215`), waehrend das `mkdir -p {log_dir}` erst in `launch.py:245`
steht, also im Skriptkoerper und damit nach dem Oeffnen der Ausgabedatei. Die
Theorie war plausibel (die beiden neuen Arme haben ein `output_dir`, das noch
nie existierte, alle Altarme haben es von frueheren Laeufen). Die
Verzeichnisse wurden von Hand angelegt - **und der naechste Job starb genauso**.
Damit ist die Theorie widerlegt, nicht bestaetigt. Sie steht hier, damit sie
niemand ein drittes Mal aufstellt.

**Echte Ursache.** `sacct --format=WorkDir` zeigt es:

```
6057297|FAILED|2:0|00:00:21|uc3n082|/tmp/duell-agent
```

Der Duell-Worktree lag unter `/tmp/duell-agent`. `/tmp` ist auf bwUniCluster
**node-lokal**, nicht geteilt. Das Verzeichnis existiert auf dem Login-Node,
auf dem Compute-Node `uc3n082` gibt es es nicht. Slurm kann nicht ins WorkDir
wechseln, bricht vor dem ersten Kommando ab und hat kein Verzeichnis, in das
es das Logfile schreiben koennte. Daher exit 2 **und** die voellige Stille -
die beiden Symptome haben dieselbe Wurzel.

**Fix.** Worktree nach `~/duell-agent` (auf `/pfs/data6`, geteilt), danach
`scripts/setup_worktree.sh` fuer data, .venv, output und external. Erster
Versuch von dort lief.

**Lehre fuers naechste Mal:** ein Job, der ohne jedes Logfile mit exit 2 in
unter 30 Sekunden stirbt, hat fast immer ein Problem *vor* dem Skript -
WorkDir, Ausgabepfad, Partition, Account. Nicht im Trainingscode suchen.
`sacct --format=JobID,State,ExitCode,Elapsed,NodeList,WorkDir%60 -P` ist der
erste Befehl, nicht der letzte.

## Nebenbefund Partitionen

- `dev_gpu_h100` schedult in unter einer Sekunde. Der in
  `PROBLEMS.md:149-152` und `scripts/slurm/README.md:127-132` beschriebene
  OpenGL-Abbruch beim Prefill auf dev ist in diesem Durchlauf **nicht**
  aufgetreten.
- `gpu_h100_short` war mit 21 fremden PENDING-Jobs belegt; ein dort
  abgesetzter Job blieb ueber Minuten auf `(Priority)` bzw. `(Resources)`.
- Konsequenz: die zwei erlaubten Parallel-Jobs auf beide Partitionen verteilen.
  dev ist der schnelle Weg zu einer Zahl, short der Reserveplatz.

## Job-Tabelle

Alle `--prod --time 00:30:00 --exclude uc3n089 --env SEED=42`.

| Job | Arm | Partition | WorkDir | Ergebnis |
|---|---|---|---|---|
| 6057269 | l3_aggregator_pooled_short | dev_gpu_h100 | /tmp/duell-agent | FAILED 2:0 nach 23 s |
| 6057270 | l3_pointmap_pose_short | gpu_h100_short | /tmp/duell-agent | gecancelt vor Start (haette dasselbe getan) |
| 6057297 | l3_aggregator_pooled_short | dev_gpu_h100 | /tmp/duell-agent | FAILED 2:0 nach 21 s |
| 6057316 | l3_aggregator_pooled_short | dev_gpu_h100 | ~/duell-agent | RUNNING ab 08:14Z auf uc3n082 |
| 6057317 | l3_pointmap_pose_short | gpu_h100_short | ~/duell-agent | PENDING ab 08:13Z |
