#!/usr/bin/env bash
# run_translator.sh
# Launch UNav Language Label Tool with diagnostics.

set -Eeuo pipefail

# -------- defaults --------
HOST_DEFAULT="127.0.0.1"
PORT_DEFAULT="5001"
USE_NAV_DEFAULT="1"
DEBUG_DEFAULT="0"
DATA_ROOT_DEFAULT="/mnt/data/UNav-IO/data"
PYTHON_BIN="${PYTHON_BIN:-python}"

# -------- logging & trap --------
log()      { echo "[translator] $*"; }
log_warn() { echo "[translator:WARN] $*" >&2; }
log_err()  { echo "[translator:ERROR] $*" >&2; }
on_err()   { log_err "Command failed (line $BASH_LINENO): $BASH_COMMAND"; exit 1; }
trap on_err ERR

usage() {
  cat <<USAGE
Usage:
  run_translator.sh [options]

Options:
  -r, --data-final-root PATH   Path to DATA_FINAL_ROOT (default: ${DATA_ROOT_DEFAULT})
  -H, --host HOST              Bind host (default: ${HOST_DEFAULT})
  -p, --port PORT              Bind port (default: ${PORT_DEFAULT}; auto-fallback if busy)
  -n, --no-nav                 Disable FacilityNavigator; use disk/files only
  -d, --debug                  Enable Flask debug (dev only)
  -v, --verbose                Print extra diagnostics
  -h, --help                   Show this help

Env:
  PYTHON_BIN=/path/to/python    Override Python interpreter

Examples:
  ./run_translator.sh
  ./run_translator.sh -r /some/other/data -H 0.0.0.0 -p 5010
USAGE
}

verbose=false
HOST="$HOST_DEFAULT"; PORT="$PORT_DEFAULT"
USE_NAV="$USE_NAV_DEFAULT"; DEBUG="$DEBUG_DEFAULT"
DATA_FINAL_ROOT="$DATA_ROOT_DEFAULT"

# -------- parse args --------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -r|--data-final-root) DATA_FINAL_ROOT="${2:-}"; shift 2 ;;
    -H|--host) HOST="${2:-}"; shift 2 ;;
    -p|--port) PORT="${2:-}"; shift 2 ;;
    -n|--no-nav) USE_NAV="0"; shift ;;
    -d|--debug) DEBUG="1"; shift ;;
    -v|--verbose) verbose=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) log_err "Unknown option: $1"; usage; exit 1 ;;
  esac
done

# -------- checks --------
if [[ ! -d "${DATA_FINAL_ROOT}" ]]; then
  log_err "DATA_FINAL_ROOT not found or not a directory: ${DATA_FINAL_ROOT}"
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  log_err "Python not found: ${PYTHON_BIN}"
  exit 1
fi

# Check flask
if ! "${PYTHON_BIN}" -c "import flask" >/dev/null 2>&1; then
  log_err "Missing Python package: Flask (pip install Flask)"
  exit 1
fi

# Check unav
if ! "${PYTHON_BIN}" -c "import unav" >/dev/null 2>&1; then
  log_err "Cannot import 'unav'. Install it into your env with:"
  log_err "  pip install -e ."
  exit 1
fi

# Ensure labels file exists
labels="${DATA_FINAL_ROOT}/_i18n/labels.json"
if [[ ! -f "$labels" ]]; then
  mkdir -p "$(dirname "$labels")"
  printf '%s\n' '{ "places": {}, "buildings": {}, "floors": {}, "destinations": {}, "aliases": {} }' > "$labels"
  $verbose && log "Initialized labels file: $labels"
fi

# Port check
port_free() {
  local host="$1" port="$2"
  if command -v nc >/dev/null 2>&1; then
    nc -z -w1 "$host" "$port" >/dev/null 2>&1 && return 1 || return 0
  elif command -v lsof >/dev/null 2>&1; then
    lsof -iTCP -sTCP:LISTEN -P -n | grep -q ":$port (LISTEN)" && return 1 || return 0
  else
    return 0
  fi
}
find_free_port() {
  local host="$1" start="$2" tries=20
  local p="$start"
  for _ in $(seq 1 "$tries"); do
    if port_free "$host" "$p"; then echo "$p"; return 0; fi
    p=$((p+1))
  done
  echo "$start"
}

PORT_SELECTED="$PORT"
if ! port_free "$HOST" "$PORT"; then
  PORT_SELECTED="$(find_free_port "$HOST" "$PORT")"
  log_warn "Port $PORT in use; switching to $PORT_SELECTED"
fi

# -------- launch --------
args=( -m unav.mapper.tools.i18n_label_web
  --data-final-root "${DATA_FINAL_ROOT}"
  --host "${HOST}"
  --port "${PORT_SELECTED}"
)
[[ "$USE_NAV" == "1" ]] && args+=( --use-nav )
[[ "$DEBUG" == "1" ]] && args+=( --debug )

log "Data root : ${DATA_FINAL_ROOT}"
log "Engine    : $([[ "$USE_NAV" == "1" ]] && echo "nav" || echo "files")"
log "URL       : http://${HOST}:${PORT_SELECTED}"

exec "${PYTHON_BIN}" "${args[@]}"
