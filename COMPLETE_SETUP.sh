#!/bin/bash
# Complete Setup Script for Production Multi-Agent RAG
# Author: Jay Guwalani

set -e

echo "=================================================="
echo "  Multi-Agent RAG System - Complete Setup"
echo "  Author: Jay Guwalani"
echo "=================================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "\n${YELLOW}[1/6]${NC} Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version detected"

# Create virtual environment
echo -e "\n${YELLOW}[2/6]${NC} Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo -e "\n${YELLOW}[3/6]${NC} Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Install core dependencies
echo -e "\n${YELLOW}[4/6]${NC} Installing core dependencies..."
cat > requirements_core.txt << EOF
# Core dependencies (REQUIRED)
langchain>=0.1.0
langchain-openai>=0.1.0
langchain-community>=0.0.20
langchain-core>=0.1.0
langgraph>=0.0.40
openai>=1.0.0
tiktoken>=0.5.0
qdrant-client>=1.7.0
pymupdf>=1.23.0
python-dotenv>=1.0.0
pydantic>=2.0.0

# API server
fastapi>=0.104.0
uvicorn>=0.24.0
requests>=2.31.0
EOF

pip install -r requirements_core.txt
echo "✓ Core dependencies installed"

# Install optional dependencies
echo -e "\n${YELLOW}[5/6]${NC} Installing optional dependencies..."
cat > requirements_optional.txt << EOF
# Performance monitoring (optional)
psutil>=5.9.0
gputil>=1.4.0

# Dashboard (optional)
dash>=2.14.0
plotly>=5.18.0

# Load testing (optional)
locust>=2.15.0
EOF

pip install -r requirements_optional.txt 2>/dev/null || echo "⚠ Some optional packages skipped"

# Setup environment
echo -e "\n${YELLOW}[6/6]${NC} Setting up environment..."

if [ ! -f ".env" ]; then
    cat > .env << EOF
# API Keys
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here

# Server Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Monitoring
ENABLE_GPU_MONITORING=true
ENABLE_SYSTEM_MONITORING=true

# Performance
CACHE_SIZE=1000
MAX_TOKENS=4096
EOF
    echo "✓ .env file created - PLEASE ADD YOUR API KEYS"
else
    echo "✓ .env file already exists"
fi

# Create start scripts
echo -e "\n${GREEN}Creating start scripts...${NC}"

# Start API server script
cat > start_api.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
python api_server.py
EOF
chmod +x start_api.sh

# Start dashboard script
cat > start_dashboard.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
python working_dashboard.py
EOF
chmod +x start_dashboard.sh

# Run tests script
cat > run_tests.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
python test_functionality.py
EOF
chmod +x run_tests.sh

# Quick demo script
cat > quick_demo.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
python production_multiagent_rag.py "What is artificial intelligence?"
EOF
chmod +x quick_demo.sh

echo "✓ Start scripts created"

# Summary
echo -e "\n=================================================="
echo -e "${GREEN}SETUP COMPLETE!${NC}"
echo "=================================================="

echo -e "\n${YELLOW}Next Steps:${NC}"
echo "1. Edit .env file and add your API keys:"
echo "   OPENAI_API_KEY=sk-..."
echo "   TAVILY_API_KEY=tvly-..."

echo -e "\n2. Run functionality tests:"
echo "   ./run_tests.sh"

echo -e "\n3. Start the system:"
echo "   Option A - API Server (for load testing):"
echo "     ./start_api.sh"
echo "   Option B - Direct CLI:"
echo "     ./quick_demo.sh"

echo -e "\n4. Start dashboard (if API running):"
echo "   ./start_dashboard.sh"

echo -e "\n${YELLOW}What's Installed:${NC}"
echo "✓ Production Multi-Agent RAG system"
echo "✓ Token optimization with cost tracking"
echo "✓ Performance monitoring"
echo "✓ REST API server"
echo "✓ Real-time dashboard"
echo "✓ Load testing support"

echo -e "\n${YELLOW}File Structure:${NC}"
echo "production_multiagent_rag.py - Main system (FULLY FUNCTIONAL)"
echo "api_server.py                - REST API server"
echo "working_dashboard.py         - Real-time dashboard"
echo "test_functionality.py        - Test suite"
echo "load_testing.py             - Load testing with Locust"

echo -e "\n${GREEN}Ready to use!${NC}"
