#!/usr/bin/env bash
# End-to-end smoke test for the LatticeDB memory engine.
#
# Verifies the full vertical slice: build flag → config parsing → registry
# lookup → factory → LatticeMemory.init → on-disk database file, by
# driving the compiled `nullclaw` binary via its `memory` subcommand.
#
# Exercised path that no unit or contract test covers:
#   CLI argv → config.Config.load → memory.initRuntime → registry.findBackend
#     → createLatticeDb → LatticeMemory.init → lattice_open() → disk
#
# Usage: scripts/e2e_latticedb.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "[e2e-latticedb] building binary with -Dengines=base,latticedb"
zig build -Dengines=base,latticedb

BIN="$REPO_ROOT/zig-out/bin/nullclaw"
if [[ ! -x "$BIN" ]]; then
  echo "[e2e-latticedb] FAIL: binary not found at $BIN" >&2
  exit 1
fi

TMP_HOME="$(mktemp -d -t nullclaw-e2e-latticedb.XXXXXX)"
trap 'rm -rf "$TMP_HOME"' EXIT

mkdir -p "$TMP_HOME/workspace"
cat > "$TMP_HOME/config.json" <<'JSON'
{
  "memory": {
    "profile": "custom",
    "backend": "latticedb",
    "auto_save": true
  }
}
JSON

export NULLCLAW_HOME="$TMP_HOME"
export NULLCLAW_WORKSPACE="$TMP_HOME/workspace"

echo "[e2e-latticedb] running: nullclaw memory stats --json"
# stats --json writes the JSON payload via std.fs.File.stdout(), so stdout
# is clean. Info logs go to stderr; suppress them to keep output readable.
STATS_JSON="$("$BIN" memory stats --json 2>/dev/null)"
echo "  -> $STATS_JSON"

if ! grep -q '"backend":"latticedb"' <<<"$STATS_JSON"; then
  echo "[e2e-latticedb] FAIL: stats did not report backend=latticedb" >&2
  echo "  got: $STATS_JSON" >&2
  exit 1
fi

# Extract entries count from stats JSON and cross-check with `memory count`.
# This proves the VTable round-trips through the CLI — stats uses the
# diagnose() path, count() uses the raw VTable.implCount() path.
STATS_ENTRIES="$(sed -n 's/.*"entries":\([0-9][0-9]*\).*/\1/p' <<<"$STATS_JSON")"
if [[ -z "$STATS_ENTRIES" ]]; then
  echo "[e2e-latticedb] FAIL: could not parse entries from stats" >&2
  exit 1
fi

echo "[e2e-latticedb] running: nullclaw memory count"
# `memory count` writes to stderr (std.debug.print), so merge streams
# and pick the lone numeric line out of the log noise.
COUNT_VAL="$("$BIN" memory count 2>&1 | grep -E '^[0-9]+$' | tail -n 1)"
echo "  -> $COUNT_VAL"
if [[ "$COUNT_VAL" != "$STATS_ENTRIES" ]]; then
  echo "[e2e-latticedb] FAIL: count=$COUNT_VAL disagrees with stats entries=$STATS_ENTRIES" >&2
  exit 1
fi

echo "[e2e-latticedb] running: nullclaw memory list --json --limit 50"
LIST_JSON="$("$BIN" memory list --json --limit 50 2>/dev/null)"
if [[ "${LIST_JSON:0:1}" != "[" ]]; then
  echo "[e2e-latticedb] FAIL: list did not return a JSON array" >&2
  echo "  got: $LIST_JSON" >&2
  exit 1
fi

# The registry resolver joins workspace_dir with "memory.db" when
# desc.needs_db_path is true. Proves the engine actually touched disk.
DB_PATH="$TMP_HOME/workspace/memory.db"
if [[ ! -e "$DB_PATH" ]]; then
  echo "[e2e-latticedb] FAIL: expected latticedb file at $DB_PATH" >&2
  ls -la "$TMP_HOME/workspace" >&2 || true
  exit 1
fi

echo "[e2e-latticedb] OK — backend initialized, stats/count healthy, db file present at $DB_PATH"
