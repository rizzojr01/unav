#!/bin/bash
# Start Floor Point Cloud Visualization Web Application
# Usage: ./run.sh [port]
# Default port: 8080

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${1:-8080}"

python3 app.py --port "$PORT"
