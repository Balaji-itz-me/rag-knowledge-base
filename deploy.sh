#!/bin/bash

# EC2 Deployment Script for RAG System
# Run this on your EC2 instance after SSH

echo "🚀 Starting RAG System Deployment on AWS EC2..."

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
echo "🐍 Installing Python and tools..."
sudo apt install python3-pip python3-venv git htop curl -y

# Clone your repository (replace with your actual repo)
echo "📥 Cloning repository..."
if [ -d "rag_demo" ]; then
    echo "Repository already exists, pulling latest changes..."
    cd rag_demo
    git pull
    cd ..
else
    git clone https://github.com/Balaji-itz-me/rag-knowledge-base.git rag_demo
fi

cd rag_demo

# Create virtual environment
echo "🔧 Setting up Python environment..."
python3 -m venv rag_env
source rag_env/bin/activate

# Install requirements
echo "📚 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create directories
echo "📁 Creating application directories..."
mkdir -p /home/ubuntu/rag_demo/{faiss_index,dynamic_index,conversations,evaluation,data}

# Set up environment variables
echo "🔑 Setting up environment variables..."
echo "Please enter your Google API key:"
read -s GOOGLE_API_KEY
export GOOGLE_API_KEY="$GOOGLE_API_KEY"

# Add to bashrc for persistence
echo "export GOOGLE_API_KEY=\"$GOOGLE_API_KEY\"" >> ~/.bashrc

# Create systemd service for auto-restart
echo "⚙️ Creating systemd service..."
sudo tee /etc/systemd/system/rag-api.service > /dev/null <<EOF
[Unit]
Description=RAG API Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/your-rag-project
Environment=GOOGLE_API_KEY=$GOOGLE_API_KEY
ExecStart=/home/ubuntu/your-rag-project/rag_env/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable rag-api.service

# Create start script for manual runs
echo "📝 Creating start script..."
cat > start_rag.sh << 'EOF'
#!/bin/bash
cd /home/ubuntu/your-rag-project
source rag_env/bin/activate
export GOOGLE_API_KEY="$GOOGLE_API_KEY"
echo "🚀 Starting RAG API server..."
echo "📡 Access at: http://$(curl -s http://checkip.amazonaws.com):8000"
echo "📚 API Docs: http://$(curl -s http://checkip.amazonaws.com):8000/docs"
python main.py
EOF

chmod +x start_rag.sh

# Get public IP
PUBLIC_IP=$(curl -s http://checkip.amazonaws.com)

echo ""
echo "✅ DEPLOYMENT COMPLETE!"
echo "===========================================" 
echo "🌐 Your RAG API is ready at:"
echo "   http://$PUBLIC_IP:8000"
echo ""
echo "📚 API Documentation:"
echo "   http://$PUBLIC_IP:8000/docs"
echo ""
echo "🎯 To start manually:"
echo "   ./start_rag.sh"
echo ""
echo "🔄 To start as service:"
echo "   sudo systemctl start rag-api.service"
echo ""
echo "📊 To check status:"
echo "   sudo systemctl status rag-api.service"
echo ""
echo "🔍 To view logs:"
echo "   sudo journalctl -u rag-api.service -f"
echo "==========================================="
