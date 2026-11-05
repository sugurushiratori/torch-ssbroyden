#!/usr/bin/env bash
# Checks that nothing internal goes public. Run before changing the repository to public
# and before every release tag (CI runs it on tags v*).  Exit code 1 lists what was found.
set -u
cd "$(git rev-parse --show-toplevel)"
fail=0
say() { echo "PUBLIC-CHECK: $*"; fail=1; }

# 1. files and folders that must stay internal (also in the history)
for p in alpha_study NOTE_for_Raj TODO_ja.md REFERENCES_ja.md lssb_bench dsgnar /tf/; do
  hits=$(git log --all --format= --name-only | grep -F "$p" | sort -u)
  [ -n "$hits" ] && say "internal path in history: $hits"
done

# 2. text in tracked files
patterns='anthropic|claude|NOTE_for_Raj|論文には書かない|\[CHECK\]|\[GIVEN\]|READ FIRST|\\todo\{|alpha.?fix|alpha_study|ss-optimistix|DSGNAR|tensorflow|gj48|gci5|/work/|/groups/|Miyabi|ABCI|repository is private'
hits=$(git grep -n -I -i -E "$patterns" -- . ':!scripts/check_public.sh' ':!.github/workflows/public-check.yml' | grep -v -E '^paper/main\.tex:[0-9]+:\\newcommand\{\\todo\}' )
[ -n "$hits" ] && say "internal text:" && echo "$hits" | head -50

# 3. commit authors, committers and messages
hits=$(git log --all --format='%h %an <%ae> | %cn <%ce>%n%B' | grep -i -E 'anthropic|claude|alpha.?fix|alpha_study|gj48|Miyabi|DSGNAR|TensorFlow|ss-optimistix' | head -20)
[ -n "$hits" ] && say "commit metadata:" && echo "$hits"

# 4. provenance metadata of binary files (images made with an assistant carry it)
for f in $(git ls-files '*.png' '*.jpg' '*.pdf'); do
  strings "$f" | grep -q -i -E 'anthropic|claude' && say "binary with assistant metadata: $f"
done

[ $fail = 0 ] && echo "PUBLIC-CHECK: OK"
exit $fail
