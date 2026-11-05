#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$DIR"

mkdir -p logs

echo "==> Stopping any previous instances on ports 8000 and 3000..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true
sleep 1

echo "==> Starting PhytoTrace FastAPI backend on http://127.0.0.1:8000..."
nohup .venv/bin/uvicorn phytotrace.api.main:app --host 127.0.0.1 --port 8000 --log-level info > logs/backend.log 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > logs/backend.pid
echo "    FastAPI PID: $BACKEND_PID"

echo "==> Starting PhytoTrace Next.js frontend on http://localhost:3000..."
cd frontend
nohup npm run start -- -p 3000 > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo $FRONTEND_PID > ../logs/frontend.pid
echo "    Next.js PID: $FRONTEND_PID"
cd ..

echo "==> Waiting for backend and frontend to become healthy..."
MAX_TRIES=20
TRIES=0
until curl -s http://127.0.0.1:8000/api/health >/dev/null 2>&1; do
    sleep 1
    TRIES=$((TRIES + 1))
    if [ $TRIES -ge $MAX_TRIES ]; then
        echo "Error: Backend failed to start. Logs:"
        cat logs/backend.log | tail -n 20
        exit 1
    fi
done
echo "    Backend is healthy at http://127.0.0.1:8000"

TRIES=0
until curl -s http://localhost:3000 >/dev/null 2>&1; do
    sleep 1
    TRIES=$((TRIES + 1))
    if [ $TRIES -ge $MAX_TRIES ]; then
        echo "Error: Frontend failed to start. Logs:"
        cat logs/frontend.log | tail -n 20
        exit 1
    fi
done
echo "    Frontend is healthy at http://localhost:3000"

echo "==> PhytoTrace AI platform is running successfully!"
echo "    Farmer & Agronomist Web UI: http://localhost:3000"
echo "    FastAPI Interactive Docs:   http://127.0.0.1:8000/docs"

