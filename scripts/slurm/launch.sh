#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: scripts/slurm/launch.sh <variant>... [--smoke | --prod | --smoke-then-prod] [--env K=V]... [--dry-run]

Variants:
  One or more config names (scripts/slurm/configs/<variant>.yaml). Bash brace
  expansion sweeps work, e.g.  launch.sh l{1,2,3,4}_vggt --smoke

Modes:
  --prod             submit the production job (default)
  --smoke            submit a short dev-cluster smoke job
  --smoke-then-prod  submit smoke, then production with an afterok dependency
  --env K=V          override a config env var (repeatable); wins over the YAML
  --dry-run          render the sbatch script(s) instead of calling sbatch
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON:-$repo_root/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
    python_bin="python"
fi

variants=()
env_args=()
mode="prod"
dry_run=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)            mode="smoke" ;;
        --prod)             mode="prod" ;;
        --smoke-then-prod)  mode="smoke-then-prod" ;;
        --dry-run)          dry_run=1 ;;
        --env)
            shift
            [[ $# -gt 0 ]] || { echo "--env requires KEY=VALUE" >&2; exit 2; }
            env_args+=(--env "$1")
            ;;
        --env=*)            env_args+=(--env "${1#--env=}") ;;
        -h|--help)          usage; exit 0 ;;
        --*)                echo "Unknown option: $1" >&2; usage; exit 2 ;;
        *)                  variants+=("$1") ;;
    esac
    shift
done

if [[ ${#variants[@]} -lt 1 ]]; then
    usage
    exit 2
fi

render() {
    local variant="$1" render_mode="$2"
    "$python_bin" scripts/slurm/launch.py "$variant" --mode "$render_mode" \
        ${env_args[@]+"${env_args[@]}"}
}

submit() {
    local variant="$1" submit_mode="$2"
    shift 2
    local script
    script="$(render "$variant" "$submit_mode")"
    local render_status=$?
    if [[ "$render_status" -ne 0 ]]; then
        return "$render_status"
    fi
    sbatch --parsable "$@" <<< "$script"
}

for variant in "${variants[@]}"; do
    case "$mode:$dry_run" in
        prod:1)
            render "$variant" prod
            ;;
        smoke:1)
            render "$variant" smoke
            ;;
        smoke-then-prod:1)
            echo "# ${variant} smoke"
            render "$variant" smoke
            echo "# ${variant} prod"
            render "$variant" prod
            ;;
        prod:0)
            prod_jid="$(submit "$variant" prod)"
            echo "[${variant} prod] jid=${prod_jid}"
            ;;
        smoke:0)
            smoke_jid="$(submit "$variant" smoke)"
            echo "[${variant} smoke] jid=${smoke_jid}"
            echo "  watch: squeue -j ${smoke_jid}"
            ;;
        smoke-then-prod:0)
            smoke_jid="$(submit "$variant" smoke)"
            prod_jid="$(submit "$variant" prod --dependency="afterok:${smoke_jid}" --kill-on-invalid-dep=yes)"
            echo "[${variant} smoke] jid=${smoke_jid}"
            echo "[${variant} prod]  jid=${prod_jid} (afterok:${smoke_jid})"
            echo "  watch: squeue -j ${smoke_jid},${prod_jid}"
            ;;
    esac
done
