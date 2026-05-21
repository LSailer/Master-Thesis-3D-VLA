#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: scripts/slurm/launch.sh <variant> [--smoke | --prod | --smoke-then-prod] [--dry-run]

Modes:
  --prod             submit the production job (default)
  --smoke            submit a dev_gpu_h100 smoke job
  --smoke-then-prod  submit smoke, then production with afterok dependency
  --dry-run          render the sbatch script instead of calling sbatch
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

if [[ $# -lt 1 ]]; then
    usage
    exit 2
fi

variant="$1"
shift
mode="prod"
dry_run=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)
            mode="smoke"
            ;;
        --prod)
            mode="prod"
            ;;
        --smoke-then-prod)
            mode="smoke-then-prod"
            ;;
        --dry-run)
            dry_run=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
    shift
done

render() {
    local render_mode="$1"
    uv run python scripts/slurm/launch.py "$variant" --mode "$render_mode"
}

submit() {
    local submit_mode="$1"
    shift
    local script
    script="$(render "$submit_mode")"
    local render_status=$?
    if [[ "$render_status" -ne 0 ]]; then
        return "$render_status"
    fi
    sbatch --parsable "$@" <<< "$script"
}

case "$mode:$dry_run" in
    prod:1)
        render prod
        ;;
    smoke:1)
        render smoke
        ;;
    smoke-then-prod:1)
        echo "# smoke"
        render smoke
        echo "# prod"
        render prod
        ;;
    prod:0)
        prod_jid="$(submit prod)"
        echo "[prod] jid=${prod_jid}"
        ;;
    smoke:0)
        smoke_jid="$(submit smoke)"
        echo "[smoke] jid=${smoke_jid}"
        echo "watch: squeue -j ${smoke_jid}"
        ;;
    smoke-then-prod:0)
        smoke_jid="$(submit smoke)"
        prod_jid="$(submit prod --dependency="afterok:${smoke_jid}" --kill-on-invalid-dep=yes)"
        echo "[smoke] jid=${smoke_jid}"
        echo "[prod]  jid=${prod_jid} (afterok:${smoke_jid})"
        echo "watch: squeue -j ${smoke_jid},${prod_jid}"
        ;;
esac
