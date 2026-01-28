#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(pwd)"
WORKTREES_DIR="${WORKTREES_DIR:-"$ROOT_DIR/.worktrees"}"
MAX_JOBS="${MAX_JOBS:-1}"

if [[ -n "$(git status --porcelain=v1)" ]]; then
  echo "Working tree is dirty. Commit/stash before running." >&2
  exit 1
fi

CURRENT_BRANCH="$(git branch --show-current)"
if [[ "$CURRENT_BRANCH" != "master" ]]; then
  echo "Expected to run on master, found $CURRENT_BRANCH." >&2
  exit 1
fi

mkdir -p "$WORKTREES_DIR"

# Default skips: tweak if you want tests for these modules too.
SKIP_FILES=(
  "gradio_app.py"
  "__init__.py"
  "rules/__init__.py"
  "taxonomy.py"
  "text.py"
  "ocr/types.py"
)

TASKS_FILE="$WORKTREES_DIR/tasks.tsv"

python3 - <<'PY' > "$TASKS_FILE"
from __future__ import annotations

from pathlib import Path

ROOT = Path(".")
SRC_ROOT = ROOT / "src" / "cola_label_verification"
TEST_ROOT = ROOT / "tests"

skip = {
    "gradio_app.py",
    "__init__.py",
    "rules/__init__.py",
    "taxonomy.py",
    "text.py",
    "ocr/types.py",
}

def expected_test_path(src_path: Path) -> Path:
    rel = src_path.relative_to(SRC_ROOT)
    name = f"test_{rel.stem}.py"
    if rel.parent == Path("."):
        return TEST_ROOT / name
    return TEST_ROOT / rel.parent / name

def branch_id(rel: Path) -> str:
    return rel.as_posix().replace("/", "_").removesuffix(".py")

modules = sorted(SRC_ROOT.rglob("*.py"))
for mod in modules:
    rel = mod.relative_to(SRC_ROOT)
    rel_posix = rel.as_posix()
    if rel_posix in skip or rel.name in skip:
        continue
    if rel.name == "__init__.py":
        continue
    test_path = expected_test_path(mod)
    if test_path.exists():
        continue
    bid = branch_id(rel)
    branch = f"codex-tests/{bid}"
    worktree = ROOT / ".worktrees" / bid
    print(
        f"{rel_posix}\t{test_path.as_posix()}\t{branch}\t{worktree.as_posix()}",
        flush=True,
    )
PY

if [[ ! -s "$TASKS_FILE" ]]; then
  echo "No missing mirrored tests found." >&2
  exit 0
fi

run_one() {
  local rel_path="$1"
  local test_path="$2"
  local branch="$3"
  local worktree_dir="$4"

  if [[ -e "$worktree_dir" ]]; then
    echo "Worktree dir already exists: $worktree_dir" >&2
    exit 1
  fi

  git worktree add -b "$branch" "$worktree_dir" >/dev/null

  local prompt
  prompt="$(cat <<EOF
You are in a git worktree of /home/james/take-home.

Target module: src/cola_label_verification/$rel_path
Target test: $test_path

Task:
- Read the target module end-to-end.
- Determine what it does, how it works, and its public surface.
- Find its callers/usages (use rg) and note their assumptions.
- Write pytest coverage that reflects real usage, edge cases, and failure modes.

Constraints:
- Create the test file at the exact target path above (mirror the src path).
- Use pytest; no new dependencies.
- Keep tests fast, deterministic, and focused.
- Do not modify files outside the target test file unless strictly required.
- Follow project conventions in AGENTS.md (PEP 8, typing, docstrings).
- If you must touch shared fixtures or helpers, keep it minimal and explain why.

Output:
- First list a short plan and the test cases you will implement.
- Then implement the tests.
EOF
)"

  (cd "$worktree_dir" && codex exec "$prompt")

  git -C "$worktree_dir" add "$test_path" >/dev/null 2>&1 || true
  if git -C "$worktree_dir" diff --cached --quiet; then
    echo "No staged changes for $rel_path" >&2
  else
    git -C "$worktree_dir" commit -m "Add tests for $rel_path" >/dev/null
  fi
}

mapfile -t TASKS < "$TASKS_FILE"

pids=()
for task in "${TASKS[@]}"; do
  IFS=$'\t' read -r rel_path test_path branch worktree_dir <<<"$task"
  run_one "$rel_path" "$test_path" "$branch" "$worktree_dir" &
  pids+=("$!")
  while (( ${#pids[@]} >= MAX_JOBS )); do
    wait -n
    live=()
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        live+=("$pid")
      fi
    done
    pids=("${live[@]}")
  done
done

wait

while IFS=$'\t' read -r rel_path test_path branch worktree_dir; do
  if git rev-list --count "master..$branch" | grep -q '[1-9]'; then
    git merge --no-ff "$branch"
  else
    echo "No commits to merge for $rel_path" >&2
  fi
done < "$TASKS_FILE"

while IFS=$'\t' read -r rel_path test_path branch worktree_dir; do
  if [[ -d "$worktree_dir" ]]; then
    if [[ -n "$(git -C "$worktree_dir" status --porcelain)" ]]; then
      echo "Dirty worktree left at $worktree_dir" >&2
      continue
    fi
    git worktree remove "$worktree_dir" >/dev/null
  fi
  if git show-ref --quiet "refs/heads/$branch"; then
    git branch -d "$branch" >/dev/null
  fi
done < "$TASKS_FILE"
