#!/bin/bash
# Bootstrap a git worktree for ML runs.
#
# Symlinks the heavy, gitignored shared dirs (`data/`, `.venv/`, `output/`)
# from the main checkout into the current worktree so that CWD-relative paths
# (e.g. Habitat dataset configs) and the shared Python env work without
# re-downloading or re-installing, and run artifacts written under `output/`
# survive the worktree being removed (they live in the main checkout).
#
# Idempotent: re-running on an already-bootstrapped worktree is a no-op
# that prints the current state of each link.
#
# Run from inside the worktree:
#   ./scripts/setup_worktree.sh

set -euo pipefail

git_common_dir="$(git rev-parse --git-common-dir)"
git_dir="$(git rev-parse --git-dir)"
if [[ "$(realpath "$git_common_dir")" == "$(realpath "$git_dir")" ]]; then
    echo "Not inside a worktree (running from the main checkout). Nothing to do."
    exit 0
fi

main_root="$(realpath "$git_common_dir/..")"
worktree_root="$(git rev-parse --show-toplevel)"

if [[ "$(realpath "$main_root")" == "$(realpath "$worktree_root")" ]]; then
    echo "Worktree root resolves to main checkout; aborting." >&2
    exit 1
fi

cd "$worktree_root"

link_shared() {
    local name="$1"
    local target_abs="$main_root/$name"

    if [[ ! -e "$target_abs" && ! -L "$target_abs" ]]; then
        echo "  - $name: skipped (not present in main checkout: $target_abs)"
        return 0
    fi

    local target_rel
    target_rel="$(realpath --relative-to="$worktree_root" "$target_abs")"

    if [[ -L "$name" ]]; then
        local current
        current="$(readlink "$name")"
        if [[ "$current" == "$target_rel" ]]; then
            echo "  - $name: ok ($current)"
            return 0
        fi
        echo "  - $name: replacing stale symlink ($current -> $target_rel)"
        rm "$name"
    elif [[ -e "$name" ]]; then
        echo "  - $name: conflict (already exists, not a symlink); leaving alone" >&2
        return 0
    fi

    ln -s "$target_rel" "$name"
    echo "  - $name: linked -> $target_rel"
}

echo "Bootstrapping worktree: $worktree_root"
echo "Main checkout:          $main_root"
echo

for name in data .venv output; do
    link_shared "$name"
done

echo
echo "Done."
