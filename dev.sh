#!/usr/bin/env bash
# Start both backend (FastAPI) and frontend (Next.js) dev servers.
# Usage: bash dev.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Kill any processes occupying ports 3000 and 8000
free_port() {
  local port=$1
  local pids
  pids=$(netstat -ano 2>/dev/null | grep ":${port} " | grep LISTENING | awk '{print $NF}' | sort -u)
  if [ -n "$pids" ]; then
    for pid in $pids; do
      echo "Killing process $pid on port $port..."
      taskkill //F //PID "$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null
    done
    sleep 1
  fi
}

echo "Clearing ports 3000 and 8000..."
free_port 3000
free_port 8000

# Remove stale Next.js dev lock file
if [ -f "$SCRIPT_DIR/frontend/.next/dev/lock" ]; then
  echo "Removing stale .next/dev/lock..."
  rm -f "$SCRIPT_DIR/frontend/.next/dev/lock"
fi

cleanup() {
  echo "Shutting down..."
  kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
  wait $BACKEND_PID $FRONTEND_PID 2>/dev/null
  exit 0
}
trap cleanup INT TERM

# Start backend
echo "Starting backend on :8000..."
cd "$SCRIPT_DIR/backend" && uv run uvicorn app.main:app --reload --port 8000 &
BACKEND_PID=$!

# Start frontend
echo "Starting frontend on :3000..."
cd "$SCRIPT_DIR/frontend" && bun dev &
FRONTEND_PID=$!

echo ""
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo "Press Ctrl+C to stop both."
echo ""

wait
