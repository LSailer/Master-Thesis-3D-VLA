#!/bin/bash
# Ableseprotokoll fuer einen Duell-Lauf. Auf dem Cluster ausfuehren.
#
#   bash readout.sh <run-dir> [baseline-metrics.csv]
#
# Setzt das Messprotokoll aus PLAN.md:58-68 um:
#   1. N = letzter geloggter Step des eigenen Laufs
#   2. eigene SR = metrics/sr an diesem Step
#   3. Baseline-SR = metrics/sr der Baseline beim naechstgelegenen Step <= N
#   4. Sekundaer: episode/steps, episode/count, perf/ms_per_step_interval
#
# Long-Format der metrics.csv ist `step,metric,value`.

set -uo pipefail

run_dir="${1:?run dir}"
baseline="${2:-$HOME/Master-Thesis-3D-VLA/output/runs/r2dreamer-curriculum-l3/run-6056750/metrics.csv}"
m="$run_dir/metrics.csv"

[[ -f "$m" ]] || { echo "FAIL: $m fehlt" >&2; exit 1; }

last_of() { grep ",$2," "$1" | tail -1 | awk -F, '{print $3}'; }

N="$(awk -F, '{print $1}' "$m" | sort -un | tail -1)"

echo "run_dir              $run_dir"
echo "N (letzter Step)     $N"
echo "metrics/sr           $(last_of "$m" metrics/sr)"
echo "metrics/sr_mean      $(last_of "$m" metrics/sr_mean)"
echo "metrics/spl          $(last_of "$m" metrics/spl)"
echo "episode/count        $(last_of "$m" episode/count)"
echo "episode/steps        $(last_of "$m" episode/steps)"
echo "metrics/dtg          $(last_of "$m" metrics/dtg)"
echo "action/forward_pct   $(last_of "$m" action/forward_pct)"
echo "action/stop_pct      $(last_of "$m" action/stop_pct)"
echo "action/left_pct      $(last_of "$m" action/left_pct)"
echo "action/right_pct     $(last_of "$m" action/right_pct)"
echo "perf/ms_per_step     $(last_of "$m" perf/ms_per_step_interval)"
echo "ms/Step (1800s/N)    $(awk -v n="$N" 'BEGIN{if(n>0) printf "%.1f", 1800/n*1000}')"

# Erfolge zaehlen: episode/success ist 1.0 bei Erfolg, 0.0 sonst.
echo "Erfolge (episode/success==1) $(awk -F, '$2=="episode/success" && $3+0==1' "$m" | wc -l | tr -d ' ')"

echo
echo "-- Baseline bei Step <= N --"
if [[ -f "$baseline" ]]; then
    awk -F, -v n="$N" '$2=="metrics/sr" && $1+0<=n {s=$1; v=$3} END{printf "Baseline-Step %s  metrics/sr %s\n", s, v}' "$baseline"
else
    echo "Baseline-CSV nicht gefunden: $baseline" >&2
fi

echo
echo "-- eigene SR-Kurve, letzte 8 Punkte --"
grep ",metrics/sr," "$m" | tail -8
