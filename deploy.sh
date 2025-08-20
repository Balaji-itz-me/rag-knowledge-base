#!/bin/bash
set -e  # Stop on errors

# Move to project folder
cd ~/rag-knowledge-base/rag_demo || { echo "❌ Project folder not found!"; exit 1; }

echo "🚀 Starting RAG System Deployment on AWS EC2..."

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
echo "🐍 Installing Python and tools..."
sudo apt install python3-pip python3-venv git htop curl -y

# Ensure repository is up-to-date
echo "📥 Ensuring repository is up-to-date..."
git pull || { echo "❌ Failed to update repo"; exit 1; }

# Create virtual environment
echo "🔧 Setting up Python environment..."
python3 -m venv rag_env
source rag_env/bin/activate

# Install requirements
if [ -f requirements.txt ]; then
    echo "📚 Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "❌ requirements.txt not found in $(pwd)! Check your folder."
    exit 1
fi

# Create application directories if missing
mkdir -p ~/rag-knowledge-base/rag_demo/{faiss_index,dynamic_index,conversations,evaluation,data,logs}

# Load environment variables from .env
if [ -f .env ]; then
    echo "🔑 Loading environment variables from .env"
    export $(grep -v '^#' .env | xargs)
else
    echo "❌ .env file not found! Please create it with GOOGLE_API_KEY before running deploy.sh"
    exit 1
fi

# Create systemd service for auto-restart
echo "⚙️ Creating systemd service..."
sudo tee /etc/systemd/system/rag-api.service > /dev/null <<EOF
[Unit]
Description=RAG API Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/rag-knowledge-base/rag_demo
Environment=GOOGLE_API_KEY=$GOOGLE_API_KEY
ExecStart=/home/ubuntu/rag-knowledge-base/rag_demo/rag_env/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable rag-api.service

# Optional: start manually with nohup
echo "📝 Creating start script..."
cat > start_rag.sh << 'EOF'
#!/bin/bash
cd ~/rag-knowledge-base/rag_demo
source rag_env/bin/activate
export $(grep -v '^#' .env | xargs)
echo "🚀 Starting RAG API server..."
nohup python main.py > logs/app.log 2>&1 &
echo "✅ Server started in background. Logs: logs/app.log"
EOF

chmod +x start_rag.sh

# Print public IP info
PUBLIC_IP=$(curl -s http://checkip.amazonaws.com)
echo ""
echo "✅ DEPLOYMENT COMPLETE!"
echo "🌐 Your RAG API is ready at: http://$PUBLIC_IP:8000"
echo "📚 API Docs: http://$PUBLIC_IP:8000/docs"
echo ""
echo "🎯 To start manually:"
echo "   ./start_rag.sh"
echo ""
echo "🔄 To manage as service:"
echo "   sudo systemctl start rag-api.service"
echo "   sudo systemctl stop rag-api.service"
echo "   sudo systemctl restart rag-api.service"
echo ""
echo "📊 To check status:"
echo "   sudo systemctl status rag-api.service"
echo ""
echo "🔍 To view logs:"
echo "   sudo journalctl -u rag-api.service -f"
echo "==========================================="
