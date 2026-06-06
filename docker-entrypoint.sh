#!/bin/bash
# docker-entrypoint.sh — Start API + Dashboard together

set -e

echo "Starting Violence Detection API on port 8000..."
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000 &

echo "Starting Streamlit Dashboard on port 8501..."
streamlit run dashboard/app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false &

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?
