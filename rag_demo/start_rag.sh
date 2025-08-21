#!/bin/bash
cd ~/rag-knowledge-base/rag_demo
source rag_env/bin/activate
export $(grep -v '^#' .env | xargs)
echo "🚀 Starting RAG API server..."
nohup python main.py > logs/app.log 2>&1 &
echo "✅ Server started in background. Logs: logs/app.log"
