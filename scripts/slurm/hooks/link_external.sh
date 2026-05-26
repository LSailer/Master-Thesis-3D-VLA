#!/usr/bin/env bash
# Symlink the heavy, gitignored external VGGT repos from the main checkout into
# the current worktree. The JAX VGGT loader resolves InfiniteVGGT/StreamVGGT/VGGT
# relative to CWD, so a job launched from a fresh worktree needs these links.
#
# Idempotent: skips links that already exist. A no-op in the main checkout
# (where target and link resolve to the same path).
set -euo pipefail

git_common_dir="$(git rev-parse --git-common-dir)"
main_root="$(realpath "${git_common_dir}/..")"

mkdir -p external
for name in InfiniteVGGT StreamVGGT VGGT; do
    target="${main_root}/external/${name}"
    link="external/${name}"
    if [ ! -e "${link}" ] && [ -e "${target}" ]; then
        ln -s "$(realpath --relative-to external "${target}")" "${link}"
        echo "linked ${link} -> $(readlink "${link}")"
    fi
done
