#!/usr/bin/env bash
# VectorSight dev launcher — strict mode
# Kills zombies, validates ports, starts backend + frontend
# Usage: bash dev.sh
set -euo pipefail

BACKEND_PORT=8003
FRONTEND_PORT=3000
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

fail() { echo -e "${RED}FATAL: $1${NC}" >&2; exit 1; }
info() { echo -e "${GREEN}[dev]${NC} $1"; }
warn() { echo -e "${YELLOW}[dev]${NC} $1"; }

# ── Step 1: Kill zombies on required ports ──────────────────────────
kill_port() {
  local port=$1
  local attempt=0
  while [ $attempt -lt 3 ]; do
    local pids
    pids=$(netstat -ano 2>/dev/null | grep ":${port} .*LISTENING" | awk '{print $5}' | sort -u | grep -v '^0$' || true)
    if [ -z "$pids" ]; then
      return 0
    fi
    for pid in $pids; do
      warn "Port ${port} occupied by PID ${pid} — killing (attempt $((attempt+1)))"
      # Kill process tree (parent + children) to avoid orphan workers
      taskkill //F //T //PID "$pid" >/dev/null 2>&1 || kill -9 "$pid" 2>/dev/null || true
    done
    # Also kill any orphan python workers still holding the port
    sleep 2
    local remaining
    remaining=$(netstat -ano 2>/dev/null | grep ":${port} .*LISTENING" | awk '{print $5}' | sort -u | grep -v '^0$' || true)
    for rpid in $remaining; do
      warn "Orphan worker PID ${rpid} still on :${port} — killing"
      taskkill //F //PID "$rpid" >/dev/null 2>&1 || true
    done
    attempt=$((attempt + 1))
    sleep 2
  done
}

info "Cleaning up ports ${BACKEND_PORT} and ${FRONTEND_PORT}..."
kill_port $BACKEND_PORT
kill_port $FRONTEND_PORT

# Remove stale Next.js lock file (prevents "is another instance running?" error)
rm -f "${FRONTEND_DIR}/.next/dev/lock" 2>/dev/null || true

# ── Step 2: Verify ports are actually free ──────────────────────────
verify_port_free() {
  local port=$1
  if netstat -ano 2>/dev/null | grep -q ":${port} .*LISTENING"; then
    fail "Port ${port} is STILL occupied after 3 attempts. Kill it manually:\n  netstat -ano | grep :${port}\n  taskkill //F //PID <pid>"
  fi
}

verify_port_free $BACKEND_PORT
verify_port_free $FRONTEND_PORT
info "Ports ${BACKEND_PORT} and ${FRONTEND_PORT} are free"

# ── Step 3: Verify .env.local points to correct backend port ───────
ENV_FILE="${FRONTEND_DIR}/.env.local"
EXPECTED_URL="http://localhost:${BACKEND_PORT}"
if [ -f "$ENV_FILE" ]; then
  CONFIGURED_URL=$(grep -oP 'NEXT_PUBLIC_API_URL=\K.*' "$ENV_FILE" 2>/dev/null || true)
  if [ "$CONFIGURED_URL" != "$EXPECTED_URL" ]; then
    warn ".env.local had ${CONFIGURED_URL:-<empty>}, fixing to ${EXPECTED_URL}"
    sed -i "s|NEXT_PUBLIC_API_URL=.*|NEXT_PUBLIC_API_URL=${EXPECTED_URL}|" "$ENV_FILE"
  fi
else
  echo "NEXT_PUBLIC_API_URL=${EXPECTED_URL}" > "$ENV_FILE"
  info "Created ${ENV_FILE}"
fi

# ── Step 4: Start backend ──────────────────────────────────────────
info "Starting backend on :${BACKEND_PORT}..."
cd "$BACKEND_DIR"
uv run uvicorn app.main:app --reload --port $BACKEND_PORT 2>&1 &
BACKEND_PID=$!

# Wait for backend health endpoint
RETRIES=0
MAX_RETRIES=30
while [ $RETRIES -lt $MAX_RETRIES ]; do
  if curl -sf "http://localhost:${BACKEND_PORT}/api/health" >/dev/null 2>&1; then
    break
  fi
  # Check if process died
  if ! kill -0 $BACKEND_PID 2>/dev/null; then
    fail "Backend process died during startup. Check logs above."
  fi
  RETRIES=$((RETRIES + 1))
  sleep 1
done
if [ $RETRIES -eq $MAX_RETRIES ]; then
  kill $BACKEND_PID 2>/dev/null || true
  fail "Backend failed to respond on :${BACKEND_PORT} after ${MAX_RETRIES}s"
fi
info "Backend healthy (PID ${BACKEND_PID})"

# ── Step 5: Start frontend ─────────────────────────────────────────
info "Starting frontend on :${FRONTEND_PORT}..."
cd "$FRONTEND_DIR"
bun dev --port $FRONTEND_PORT 2>&1 &
FRONTEND_PID=$!

# Wait for frontend to respond
RETRIES=0
while [ $RETRIES -lt $MAX_RETRIES ]; do
  if curl -sf "http://localhost:${FRONTEND_PORT}" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    kill $BACKEND_PID 2>/dev/null || true
    fail "Frontend process died during startup. Check logs above."
  fi
  RETRIES=$((RETRIES + 1))
  sleep 1
done
if [ $RETRIES -eq $MAX_RETRIES ]; then
  kill $FRONTEND_PID 2>/dev/null || true
  kill $BACKEND_PID 2>/dev/null || true
  fail "Frontend failed to respond on :${FRONTEND_PORT} after ${MAX_RETRIES}s"
fi
info "Frontend ready (PID ${FRONTEND_PID})"

# ── Step 6: Verify full proxy chain ────────────────────────────────
sleep 2
HEALTH=$(curl -sf "http://localhost:${FRONTEND_PORT}/api/health" 2>&1 || true)
if echo "$HEALTH" | grep -q '"status":"ok"'; then
  info "Proxy chain verified (frontend:${FRONTEND_PORT} → backend:${BACKEND_PORT})"
else
  warn "Proxy chain not yet responding — may need a moment for route compilation"
fi

echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}  VectorSight dev servers running${NC}"
echo -e "  Backend:  ${GREEN}http://localhost:${BACKEND_PORT}${NC}"
echo -e "  Frontend: ${GREEN}http://localhost:${FRONTEND_PORT}${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Press Ctrl+C to stop both servers"

# ── Trap: clean shutdown ───────────────────────────────────────────
cleanup() {
  echo ""
  info "Shutting down..."
  kill $FRONTEND_PID 2>/dev/null || true
  kill $BACKEND_PID 2>/dev/null || true
  wait 2>/dev/null || true
  info "Done"
}
trap cleanup EXIT INT TERM

wait
