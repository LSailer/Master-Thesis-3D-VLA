#!/bin/bash
# Duell-Gate: prueft, dass die eingefrorene Zone unberuehrt ist.
#
# Aufruf (aus dem Repo-Root):
#   bash prototyp/duell-vggt-integration/verify.sh
#   bash prototyp/duell-vggt-integration/verify.sh --record   # einmalig vor dem Duell
#
# --record schreibt die Pruefsumme des Curriculum-JSON nach
# expected_curriculum.sha256. Muss auf dem Cluster laufen, weil data/
# gitignored ist und nur dort existiert.

set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

FROZEN=(
    "src/environments"
    "data/curriculum"
    "src/shared/wandb_utils.py"
)
CURRICULUM="data/curriculum/level3_10houses_1goal.json"
SHA_FILE="$here/expected_curriculum.sha256"
BASE_REF="${DUEL_BASE_REF:-main}"

sha_of() {
    if command -v sha256sum &>/dev/null; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

if [[ "${1:-}" == "--record" ]]; then
    if [[ ! -f "$CURRICULUM" ]]; then
        echo "FAIL: $CURRICULUM nicht gefunden (laeuft dieses Script auf dem Cluster?)" >&2
        exit 1
    fi
    sha_of "$CURRICULUM" > "$SHA_FILE"
    echo "Pruefsumme aufgezeichnet: $(cat "$SHA_FILE")"
    echo "  -> $SHA_FILE"
    exit 0
fi

fail=0

echo "== 1. Eingefrorene Pfade (Diff gegen $BASE_REF) =="
if ! git rev-parse --verify --quiet "$BASE_REF" >/dev/null; then
    echo "  FAIL: Referenz '$BASE_REF' existiert nicht" >&2
    fail=1
else
    base="$(git merge-base "$BASE_REF" HEAD)"
    touched="$(git diff --name-only "$base" HEAD -- "${FROZEN[@]}")"
    uncommitted="$(git status --porcelain -- "${FROZEN[@]}")"
    if [[ -n "$touched" ]]; then
        echo "  FAIL: committete Aenderungen in der eingefrorenen Zone:" >&2
        echo "$touched" | sed 's/^/    /' >&2
        fail=1
    fi
    if [[ -n "$uncommitted" ]]; then
        echo "  FAIL: uncommittete Aenderungen in der eingefrorenen Zone:" >&2
        echo "$uncommitted" | sed 's/^/    /' >&2
        fail=1
    fi
    [[ -z "$touched" && -z "$uncommitted" ]] && echo "  OK: unberuehrt"
fi

echo "== 2. Curriculum-Pruefsumme =="
if [[ ! -f "$CURRICULUM" ]]; then
    echo "  SKIP: $CURRICULUM nicht vorhanden (nicht auf dem Cluster?)"
elif [[ ! -f "$SHA_FILE" ]]; then
    echo "  SKIP: keine Referenz-Pruefsumme, erst 'verify.sh --record' laufen lassen"
else
    actual="$(sha_of "$CURRICULUM")"
    expected="$(cat "$SHA_FILE")"
    if [[ "$actual" != "$expected" ]]; then
        echo "  FAIL: Curriculum veraendert" >&2
        echo "    erwartet: $expected" >&2
        echo "    ist:      $actual" >&2
        fail=1
    else
        echo "  OK: unveraendert"
    fi
fi

echo "== 3. Seed =="
configs="$(git diff --name-only "$(git merge-base "$BASE_REF" HEAD)" HEAD -- 'scripts/slurm/configs/*.yaml'; git status --porcelain --short -- 'scripts/slurm/configs/*.yaml' | awk '{print $2}')"
configs="$(echo "$configs" | sort -u | sed '/^$/d')"
if [[ -z "$configs" ]]; then
    echo "  SKIP: keine geaenderte oder neue SLURM-Config gefunden"
else
    for cfg in $configs; do
        [[ -f "$cfg" ]] || continue
        if grep -qE 'SEED:[[:space:]]*"?42"?' "$cfg"; then
            echo "  OK: $cfg -> SEED=42"
        else
            echo "  FAIL: $cfg setzt SEED nicht auf 42" >&2
            grep -nE 'seed|SEED' "$cfg" | sed 's/^/    /' >&2
            fail=1
        fi
    done
fi

echo
if [[ $fail -ne 0 ]]; then
    echo "VERIFY: FAIL"
    exit 1
fi
echo "VERIFY: PASS"
