#!/usr/bin/env bash
#
# Stop everything this project started.
#
#   ./scripts/stop.sh                # stop API, UI and containers; keep data
#   ./scripts/stop.sh --volumes      # also delete the OpenSearch index volume
#   ./scripts/stop.sh --all          # volumes + venv + node_modules + generated data
#   ./scripts/stop.sh --status       # show what is running, stop nothing
#
# Safe to run repeatedly and safe to run when nothing is up.
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$PWD"
RUN_DIR="$ROOT/.run"

DROP_VOLUMES=0 DROP_ALL=0 STATUS_ONLY=0
API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-5173}"
OS_PORT="${OS_PORT:-9200}"

if [[ -t 1 ]]; then
  B=$'\033[1m'; BLUE=$'\033[34m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; R=$'\033[0m'
else
  B=""; BLUE=""; GREEN=""; YELLOW=""; R=""
fi
step() { printf '\n%s%s==>%s %s\n' "$B" "$BLUE" "$R" "$1"; }
ok()   { printf '    %s✓%s %s\n' "$GREEN" "$R" "$1"; }
skip() { printf '    %s-%s %s\n' "$YELLOW" "$R" "$1"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --volumes) DROP_VOLUMES=1; shift ;;
    --all)     DROP_VOLUMES=1; DROP_ALL=1; shift ;;
    --status)  STATUS_ONLY=1; shift ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) printf 'unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done

COMPOSE_CMD=""
if docker compose version >/dev/null 2>&1; then COMPOSE_CMD="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then COMPOSE_CMD="docker-compose"; fi

listening() { command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ":$1 "; }

# ------------------------------------------------------------------- status
if [[ "$STATUS_ONLY" -eq 1 ]]; then
  step "Status"
  for svc in api ui; do
    f="$RUN_DIR/$svc.pid"
    if [[ -f "$f" ]] && kill -0 "$(cat "$f")" 2>/dev/null; then
      ok "$svc running (pid $(cat "$f"))"
    else
      skip "$svc not running"
    fi
  done
  for p in "$API_PORT:api" "$UI_PORT:ui" "$OS_PORT:opensearch"; do
    port="${p%%:*}"; name="${p##*:}"
    listening "$port" && ok "port $port ($name) in use" || skip "port $port ($name) free"
  done
  if [[ -n "$COMPOSE_CMD" ]]; then
    running=$($COMPOSE_CMD ps --services --filter status=running 2>/dev/null || true)
    [[ -n "$running" ]] && ok "containers: $(echo "$running" | tr '\n' ' ')" \
                        || skip "no containers running"
  fi
  exit 0
fi

# --------------------------------------------------- local background procs
step "Stopping local processes"
for svc in api ui; do
  f="$RUN_DIR/$svc.pid"
  if [[ -f "$f" ]]; then
    pid=$(cat "$f")
    if kill -0 "$pid" 2>/dev/null; then
      # SIGTERM first so uvicorn/vite can close sockets cleanly.
      kill "$pid" 2>/dev/null || true
      for _ in $(seq 1 10); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
      if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null || true
        ok "$svc force-killed (pid $pid)"
      else
        ok "$svc stopped (pid $pid)"
      fi
    else
      skip "$svc pidfile stale"
    fi
    rm -f "$f"
  else
    skip "$svc not started by this project"
  fi
done

# Vite spawns children that can outlive the parent; clear the port if still held.
for p in "$UI_PORT:ui" "$API_PORT:api"; do
  port="${p%%:*}"; name="${p##*:}"
  if listening "$port"; then
    pids=$(ss -ltnp 2>/dev/null | awk -v P=":$port " '$4 ~ P {print $NF}' \
           | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u || true)
    for pid in $pids; do
      # Only kill processes rooted in this project, never something unrelated.
      if [[ -r "/proc/$pid/cwd" ]] && [[ "$(readlink -f "/proc/$pid/cwd" 2>/dev/null)" == "$ROOT"* ]]; then
        kill "$pid" 2>/dev/null || true
        ok "released port $port ($name, pid $pid)"
      else
        skip "port $port held by a process outside this project (pid $pid) - left alone"
      fi
    done
  fi
done

# ------------------------------------------------------------- containers
step "Stopping containers"
if [[ -z "$COMPOSE_CMD" ]]; then
  skip "docker compose not available"
elif ! docker info >/dev/null 2>&1; then
  skip "docker daemon not reachable"
else
  if [[ "$DROP_VOLUMES" -eq 1 ]]; then
    $COMPOSE_CMD down -v --remove-orphans
    ok "containers stopped, volumes removed (the index is gone; rebuild with setup.sh)"
  else
    $COMPOSE_CMD down --remove-orphans
    ok "containers stopped, volumes kept (the index survives)"
  fi
fi

# ------------------------------------------------------------------ purge
if [[ "$DROP_ALL" -eq 1 ]]; then
  step "Removing build artefacts"
  for p in .venv ui/node_modules ui/dist data/processed data/ingestion.db \
           .pytest_cache .run reports; do
    if [[ -e "$p" ]]; then rm -rf "$p"; ok "removed $p"; fi
  done
  find . -type d -name __pycache__ -not -path './.git/*' -exec rm -rf {} + 2>/dev/null || true
  ok "removed __pycache__"
  skip "kept: .env, patent_data/, data/evaluation/ (judgements are expensive to rebuild)"
fi

step "Done"
printf '\n  Start again with:  ./scripts/setup.sh\n'
printf '  Check status with: ./scripts/stop.sh --status\n\n'
