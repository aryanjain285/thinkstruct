#!/usr/bin/env bash
#
# Full A-Z bootstrap for macOS or Linux with Docker.
#
#   ./scripts/setup.sh                       # everything: deps, index, API, UI
#   ./scripts/setup.sh --embedder openai-small   # hosted embeddings (needs OPENAI_API_KEY)
#   ./scripts/setup.sh --no-ui               # skip the React frontend
#   ./scripts/setup.sh --docker              # run API + UI in containers instead of locally
#   ./scripts/setup.sh --clean               # tear everything down first
#
# Defaults to sentence-transformers running locally, so no API key is required.
set -Eeuo pipefail

# ------------------------------------------------------------------ settings
EMBEDDER="${PATSEARCH_EMBEDDER:-minilm}"
DO_INDEX=1 DO_TESTS=1 DO_UI=1 DO_API=1 USE_DOCKER=0 DO_CLEAN=0
API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-5173}"
OS_PORT="${OS_PORT:-9200}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$PWD"
RUN_DIR="$ROOT/.run"
mkdir -p "$RUN_DIR"

# ------------------------------------------------------------------- output
if [[ -t 1 ]]; then
  B=$'\033[1m'; BLUE=$'\033[34m'; GREEN=$'\033[32m'; RED=$'\033[31m'
  YELLOW=$'\033[33m'; R=$'\033[0m'
else
  B=""; BLUE=""; GREEN=""; RED=""; YELLOW=""; R=""
fi
step() { printf '\n%s%s==>%s %s%s\n' "$B" "$BLUE" "$R" "$1" "$R"; }
ok()   { printf '    %s✓%s %s\n' "$GREEN" "$R" "$1"; }
warn() { printf '    %s!%s %s\n' "$YELLOW" "$R" "$1"; }
die()  { printf '\n%sERROR:%s %s\n' "$RED" "$R" "$1" >&2; exit 1; }

trap 'die "failed at line $LINENO. Re-run with: bash -x $0"' ERR

# --------------------------------------------------------------------- args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --embedder)  EMBEDDER="$2"; shift 2 ;;
    --no-index)  DO_INDEX=0; shift ;;
    --no-tests)  DO_TESTS=0; shift ;;
    --no-ui)     DO_UI=0; shift ;;
    --no-api)    DO_API=0; shift ;;
    --docker)    USE_DOCKER=1; shift ;;
    --clean)     DO_CLEAN=1; shift ;;
    -h|--help)   sed -n '2,12p' "$0"; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

# ------------------------------------------------------------------- helpers
port_free() {
  local port=$1
  if command -v lsof >/dev/null 2>&1; then
    ! lsof -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null | grep -q .
  elif command -v ss >/dev/null 2>&1; then
    ! ss -ltn 2>/dev/null | grep -q ":$port "
  elif command -v netstat >/dev/null 2>&1; then
    ! netstat -an 2>/dev/null | grep -E "[.:]$port[[:space:]].*LISTEN" >/dev/null
  else
    warn "cannot verify whether port $port is free; continuing"
    return 0
  fi
}

inplace_sed() {
  local expression=$1 file=$2
  if [[ "$(uname -s)" == "Darwin" ]]; then
    sed -i '' "$expression" "$file"
  else
    sed -i "$expression" "$file"
  fi
}

stop_pidfile() {
  local f="$RUN_DIR/$1.pid"
  [[ -f "$f" ]] || return 0
  local p; p=$(cat "$f")
  if kill -0 "$p" 2>/dev/null; then kill "$p" 2>/dev/null || true; sleep 1; fi
  rm -f "$f"
}

wait_http() {  # wait_http <url> <seconds> <label>
  local url=$1 secs=$2 label=$3 i
  printf '    waiting for %s' "$label"
  for ((i=0; i<secs; i++)); do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then printf '\n'; return 0; fi
    printf '.'; sleep 1
  done
  printf '\n'; return 1
}

# ------------------------------------------------------------- prerequisites
step "Checking prerequisites"

command -v curl >/dev/null || die "curl not found. sudo apt-get install -y curl"
ok "curl"

command -v docker >/dev/null || die "docker not found. See https://docs.docker.com/engine/install/"
docker info >/dev/null 2>&1 || die "cannot reach the Docker daemon. Start it, and ensure your user is in the 'docker' group (newgrp docker)."
ok "docker $(docker --version | awk '{print $3}' | tr -d ,)"

if docker compose version >/dev/null 2>&1; then COMPOSE_CMD="docker compose"
elif command -v docker-compose >/dev/null;   then COMPOSE_CMD="docker-compose"
else die "docker compose not available."; fi
ok "compose ($COMPOSE_CMD)"

# --clean is handled here, once COMPOSE_CMD is known.
if [[ "$DO_CLEAN" -eq 1 ]]; then
  step "Tearing down"
  stop_pidfile api; stop_pidfile ui
  $COMPOSE_CMD down -v 2>/dev/null || true
  rm -rf .venv ui/node_modules ui/dist data/processed data/ingestion.db
  ok "cleaned"
fi

PY=""
for c in python3.12 python3.11 python3; do
  command -v "$c" >/dev/null || continue
  if [[ "$("$c" -c 'import sys;print(sys.version_info[0]*100+sys.version_info[1])')" -ge 311 ]]; then
    PY="$c"; break
  fi
done
[[ -n "$PY" ]] || die "need Python 3.11+. sudo apt-get install -y python3.12 python3.12-venv"
"$PY" -c 'import venv' 2>/dev/null || die "python venv module missing. sudo apt-get install -y ${PY}-venv"
ok "$PY $("$PY" --version | awk '{print $2}')"

if [[ "$DO_UI" -eq 1 ]]; then
  if command -v node >/dev/null && [[ "$(node -v | tr -d 'v' | cut -d. -f1)" -ge 18 ]]; then
    ok "node $(node -v)"
  elif [[ "$USE_DOCKER" -eq 1 ]]; then
    ok "node not needed (UI builds in Docker)"
  else
    warn "node 18+ not found — skipping the UI. Install it or pass --docker."
    DO_UI=0
  fi
fi

# OpenSearch runs in Linux. On macOS, inspect Docker Desktop's Linux VM.
if [[ "$(uname -s)" == "Darwin" ]]; then
  MMC=$(docker run --rm --privileged --pid=host alpine     sysctl -n vm.max_map_count 2>/dev/null || echo 0)

  if [[ "$MMC" -lt 262144 ]]; then
    warn "Docker VM vm.max_map_count=$MMC, OpenSearch needs >= 262144"
    die "run this first:
docker run --rm --privileged --pid=host alpine \
  sysctl -w vm.max_map_count=262144"
  fi

  ok "Docker VM vm.max_map_count=$MMC"
else
  MMC=$(sysctl -n vm.max_map_count 2>/dev/null || echo 0)

  if [[ "$MMC" -lt 262144 ]]; then
    warn "vm.max_map_count=$MMC, OpenSearch needs >= 262144"
    if sudo -n true 2>/dev/null; then
      sudo sysctl -w vm.max_map_count=262144 >/dev/null
      ok "raised vm.max_map_count"
    else
      die "run this first:
sudo sysctl -w vm.max_map_count=262144

To persist it:
echo 'vm.max_map_count=262144' | sudo tee -a /etc/sysctl.conf"
    fi
  else
    ok "vm.max_map_count=$MMC"
  fi
fi

# --------------------------------------------------------------------- data
step "Checking the corpus"

# Default corpus location:
#   <repository>/data/patent_data_small
#
# Override when needed:
#   PATSEARCH_DATA_DIR=/absolute/path/to/patent_data_small ./scripts/setup.sh
DATA_DIR="${PATSEARCH_DATA_DIR:-$ROOT/data/patent_data_small}"

if [[ ! -d "$DATA_DIR" ]]; then
  ZIP=$(find "$ROOT" -maxdepth 1 -iname '*patent_data*.zip' | head -1 || true)

  if [[ -z "$ZIP" ]]; then
    die "corpus missing.

Expected directory:
  $DATA_DIR

Your corpus should contain files named patents_*.json.

You can also provide a custom location:
  PATSEARCH_DATA_DIR=/absolute/path/to/patent_data_small ./scripts/setup.sh"
  fi

  command -v unzip >/dev/null || die "unzip not found. Install it and re-run."

  mkdir -p "$ROOT/data"
  unzip -q -o "$ZIP" -d "$ROOT/data"
  ok "extracted $(basename "$ZIP") into $ROOT/data"

  # Some archives contain an extra data/ directory.
  if [[ ! -d "$DATA_DIR" && -d "$ROOT/data/data/patent_data_small" ]]; then
    DATA_DIR="$ROOT/data/data/patent_data_small"
  fi
fi

NFILES=$(find "$DATA_DIR" -type f -name 'patents_*.json' | wc -l | tr -d ' ')
[[ "$NFILES" -gt 0 ]] || die "no patents_*.json files found under:
  $DATA_DIR"

ok "$NFILES weekly JSON files in $DATA_DIR"

# ------------------------------------------------------------------- python
step "Python environment"
if [[ ! -d .venv ]]; then
  if command -v uv >/dev/null; then uv venv .venv --python 3.12 >/dev/null
  else "$PY" -m venv .venv; fi
fi
VPY="$ROOT/.venv/bin/python"
[[ -x "$VPY" ]] || die ".venv was not created properly"

if command -v uv >/dev/null; then
  uv pip install -e ".[all]" --python "$VPY" --quiet
else
  "$VPY" -m pip install --upgrade pip --quiet
  "$VPY" -m pip install -e ".[all]" --quiet
fi
ok "dependencies installed ($("$VPY" -m pip list 2>/dev/null | wc -l) packages)"

# --------------------------------------------------------------------- .env
step "Configuration"
[[ -f .env ]] || { cp .env.example .env; ok "created .env from template"; }
if grep -q '^PATSEARCH_EMBEDDER=' .env; then
  inplace_sed "s|^PATSEARCH_EMBEDDER=.*|PATSEARCH_EMBEDDER=$EMBEDDER|" .env
else
  echo "PATSEARCH_EMBEDDER=$EMBEDDER" >> .env
fi
ok "embedder = $EMBEDDER"

if [[ "$EMBEDDER" == openai-* ]]; then
  grep -qE '^OPENAI_API_KEY=sk-' .env \
    || die "'$EMBEDDER' is a hosted model but OPENAI_API_KEY is not set in .env"
  ok "hosted embeddings configured"
else
  step "Downloading the embedding model (local inference, no API key)"
  SETUP_EMBEDDER="$EMBEDDER" "$VPY" - <<'PYEOF'
import os, sys
sys.path.insert(0, "src")
from patsearch.embeddings.service import create_embedder
e = create_embedder(os.environ.get("SETUP_EMBEDDER", "minilm"))
print(f"    {e.model_name}  dim={e.dimension}")
PYEOF
  ok "model cached locally"
fi

# --------------------------------------------------------------- opensearch
step "Starting OpenSearch (Docker — nothing to install by hand)"

if ! docker image inspect opensearchproject/opensearch:3.7.0 >/dev/null 2>&1; then
  echo "    pulling opensearchproject/opensearch:3.7.0 (~1 GB, first run only)"
  $COMPOSE_CMD pull opensearch || die "image pull failed. Check network/proxy."
  ok "image pulled"
else
  ok "image already present"
fi

$COMPOSE_CMD up -d opensearch

# The JVM plus 25 plugins take a while on a cold start; be generous.
if ! wait_http "http://localhost:$OS_PORT/_cluster/health" 180 "cluster health"; then
  echo
  echo "    last 30 lines of the container log:"
  $COMPOSE_CMD logs --tail 30 opensearch 2>&1 | sed 's/^/      /'
  die "OpenSearch did not become healthy.
   Most common causes:
     - vm.max_map_count too low:
         Linux: sudo sysctl -w vm.max_map_count=262144
         macOS: docker run --rm --privileged --pid=host alpine sysctl -w vm.max_map_count=262144
     - not enough Docker memory (allocate at least ~4 GB)
     - port $OS_PORT already in use"
fi
ok "cluster $(curl -s "http://localhost:$OS_PORT/_cluster/health" | grep -o '"status":"[a-z]*"' | cut -d: -f2)"
ok "OpenSearch on http://localhost:$OS_PORT"

# -------------------------------------------------------------------- tests
if [[ "$DO_TESTS" -eq 1 ]]; then
  step "Running the test suite"
  "$VPY" -m pytest tests/ -q --ignore=tests/test_integration.py
  ok "unit tests passed"
fi

# -------------------------------------------------------------------- index
if [[ "$DO_INDEX" -eq 1 ]]; then
  step "Building the index"
  "$VPY" scripts/build_index.py --embeddings --embedder "$EMBEDDER"
  ok "$(curl -s "http://localhost:$OS_PORT/patents/_count" | grep -o '"count":[0-9]*') records"

  if [[ "$DO_TESTS" -eq 1 ]]; then
    step "Running integration tests against the live index"
    "$VPY" -m pytest tests/test_integration.py -q
    ok "integration tests passed"
  fi
fi

# ---------------------------------------------------------------- services
API_URL="${API_URL:-http://localhost:$API_PORT}"
UI_URL="${UI_URL:-}"

if [[ "$USE_DOCKER" -eq 1 ]]; then
  step "Starting API and UI in Docker"
  $COMPOSE_CMD up -d --build api ui
  wait_http "http://localhost:8000/health" 180 "API" || die "API did not start. $COMPOSE_CMD logs api"
  ok "API   http://localhost:8000"
  wait_http "http://localhost:8080" 60 "UI" || warn "UI not responding yet"
  ok "UI    http://localhost:8080"
  API_URL="http://localhost:8000"; UI_URL="http://localhost:8080"
else
  if [[ "$DO_API" -eq 1 ]]; then
    step "Starting the API"
    stop_pidfile api
    port_free "$API_PORT" || die "port $API_PORT is in use. Set API_PORT=... or free it."
    PYTHONPATH="$ROOT/src" nohup "$ROOT/.venv/bin/uvicorn" patsearch.api.main:app \
      --host 0.0.0.0 --port "$API_PORT" > "$RUN_DIR/api.log" 2>&1 &
    echo $! > "$RUN_DIR/api.pid"
    wait_http "http://localhost:$API_PORT/health" 120 "API" \
      || die "API failed to start. Log: $RUN_DIR/api.log"
    ok "API   http://localhost:$API_PORT   (log: .run/api.log)"
    API_URL="http://localhost:$API_PORT"
  fi

  if [[ "$DO_UI" -eq 1 ]]; then
    step "Building and starting the UI"
    stop_pidfile ui
    port_free "$UI_PORT" || die "port $UI_PORT is in use. Set UI_PORT=... or free it."
    ( cd ui && npm install --silent )
    ( cd ui && VITE_API_URL="$API_URL" nohup npm run dev -- --port "$UI_PORT" --host \
        > "$RUN_DIR/ui.log" 2>&1 & echo $! > "$RUN_DIR/ui.pid" )
    wait_http "http://localhost:$UI_PORT" 90 "UI" || warn "UI slow to start; check .run/ui.log"
    ok "UI    http://localhost:$UI_PORT   (log: .run/ui.log)"
    UI_URL="http://localhost:$UI_PORT"
  fi
fi

# ------------------------------------------------------------------- finish
step "Ready"
cat <<EOF

  ${B}Open the app:${R}  ${UI_URL:-（ui not started）}
  ${B}API docs:${R}      ${API_URL:-http://localhost:$API_PORT}/docs

  ${B}CLI search${R}
    .venv/bin/python scripts/search.py --method hybrid \\
      --query "flexible fibre spoke connected between a hub and a wheel rim" \\
      --classification-prefix B60B

  ${B}Latency, filters on vs off${R}     (Part 1 deliverable)
    .venv/bin/python scripts/benchmark.py --methods bm25 dense hybrid

  ${B}Evaluation${R}                     (Part 3 deliverable)
    .venv/bin/python scripts/evaluate.py --generate
    .venv/bin/python scripts/evaluate.py --systems bm25 dense hybrid --baseline bm25

  ${B}Ingestion queue POC${R}            (Part 2 deliverable)
    .venv/bin/python scripts/ingest.py --enqueue --run

  ${B}Stop everything${R}
    ./scripts/setup.sh --clean

EOF
