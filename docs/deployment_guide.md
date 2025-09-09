# Deployment Guide - Multi-Agent RAG System

**Author:** Jay Guwalani  
**Date:** December 2024

## Overview

This guide covers deployment strategies for the Multi-Agent RAG System in various environments, from local development to production cloud deployments.

## Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: 2GB available space
- **Network**: Internet connection for API calls

### API Requirements

- **OpenAI API Key**: GPT-4 access recommended
- **Tavily API Key**: For web search functionality

## Local Development Setup

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/JayDS22/multi-agent-rag-system.git
cd multi-agent-rag-system

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your API keys
nano .env
```

### 3. Verification

```python
# Test installation
python -c "from multi_agent_rag import MultiAgentRAGSystem; print('Installation successful!')"

# Run example
python examples/research_example.py
```

## Docker Deployment

### 1. Dockerfile

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port (if running web service)
EXPOSE 8000

# Command to run application
CMD ["python", "-m", "multi_agent_rag"]
```

### 2. Docker Compose

```yaml
version: '3.8'

services:
  multi-agent-rag:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - TAVILY_API_KEY=${TAVILY_API_KEY}
    volumes:
      - ./output:/app/output
    ports:
      - "8000:8000"
    restart: unless-stopped
    
  # Optional: Redis for caching
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
```

### 3. Build and Run

```bash
# Build image
docker build -t multi-agent-rag .

# Run with environment variables
docker run -e OPENAI_API_KEY=your_key -e TAVILY_API_KEY=your_key multi-agent-rag

# Or use docker-compose
docker-compose up -d
```

## Cloud Deployment

### AWS Deployment

#### 1. EC2 Instance Setup

```bash
# Launch EC2 instance (t3.medium or larger recommended)
# Connect via SSH

# Update system
sudo yum update -y

# Install Python 3.11
sudo amazon-linux-extras install python3.11

# Install Git
sudo yum install git -y

# Clone and setup application
git clone https://github.com/JayDS22/multi-agent-rag-system.git
cd multi-agent-rag-system

# Setup virtual environment
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 2. AWS Lambda Deployment

```python
# lambda_function.py
import json
import os
from multi_agent_rag import MultiAgentRAGSystem

def lambda_handler(event, context):
    """AWS Lambda handler for Multi-Agent RAG System."""
    
    # Initialize system
    system = MultiAgentRAGSystem(
        openai_key=os.environ['OPENAI_API_KEY'],
        tavily_key=os.environ['TAVILY_API_KEY']
    )
    
    # Process request
    message = event.get('message', '')
    mode = event.get('mode', 'full')  # full, research, document
    
    try:
        if mode == 'research':
            results = system.research_only(message)
        elif mode == 'document':
            results = system.document_only(message)
        else:
            results = system.process_request(message)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'success': True,
                'results': results
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'success': False,
                'error': str(e)
            })
        }
```

#### 3. SAM Template

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Globals:
  Function:
    Timeout: 300
    MemorySize: 1024

Resources:
  MultiAgentRAGFunction:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: src/
      Handler: lambda_function.lambda_handler
      Runtime: python3.11
      Environment:
        Variables:
          OPENAI_API_KEY: !Ref OpenAIAPIKey
          TAVILY_API_KEY: !Ref TavilyAPIKey
      Events:
        MultiAgentRAGAPI:
          Type: Api
          Properties:
            Path: /process
            Method: post

Parameters:
  OpenAIAPIKey:
    Type: String
    NoEcho: true
  TavilyAPIKey:
    Type: String
    NoEcho: true
```

### Google Cloud Platform

#### 1. Cloud Run Deployment

```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/multi-agent-rag', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/multi-agent-rag']
  - name: 'gcr.io/cloud-builders/gcloud'
    args: [
      'run', 'deploy', 'multi-agent-rag',
      '--image', 'gcr.io/$PROJECT_ID/multi-agent-rag',
      '--platform', 'managed',
      '--region', 'us-central1',
      '--allow-unauthenticated'
    ]
```

#### 2. Deployment Commands

```bash
# Build and deploy to Cloud Run
gcloud builds submit --config cloudbuild.yaml

# Set environment variables
gcloud run services update multi-agent-rag \
    --set-env-vars OPENAI_API_KEY=your_key,TAVILY_API_KEY=your_key \
    --region us-central1
```

### Azure Deployment

#### 1. Container Instance

```bash
# Create resource group
az group create --name MultiAgentRAG --location eastus

# Create container instance
az container create \
    --resource-group MultiAgentRAG \
    --name multi-agent-rag \
    --image your-registry/multi-agent-rag:latest \
    --cpu 2 \
    --memory 4 \
    --environment-variables \
        OPENAI_API_KEY=your_key \
        TAVILY_API_KEY=your_key \
    --ports 8000
```

## Production Considerations

### 1. Environment Variables

```bash
# Production environment setup
export OPENAI_API_KEY="your_production_key"
export TAVILY_API_KEY="your_production_key"
export ENVIRONMENT="production"
export LOG_LEVEL="INFO"
export MAX_WORKERS="4"
export CACHE_TTL="3600"
```

### 2. Security Configuration

```python
# security_config.py
import os

# API rate limiting
RATE_LIMIT_PER_MINUTE = 60
RATE_LIMIT_PER_HOUR = 1000

# Authentication
REQUIRE_API_KEY = True
API_KEY_HEADER = "X-API-Key"

# Logging
LOG_SENSITIVE_DATA = False
LOG_RETENTION_DAYS = 30

# Resource limits
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_PROCESSING_TIME = 300  # 5 minutes
MAX_CONCURRENT_REQUESTS = 10
```

### 3. Monitoring Setup

```python
# monitoring.py
import logging
import time
from functools import wraps

def monitor_performance(func):
    """Decorator to monitor function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            logging.info(f"{func.__name__} completed in {time.time() - start_time:.2f}s")
            return result
        except Exception as e:
            logging.error(f"{func.__name__} failed: {str(e)}")
            raise
    return wrapper

# Health check endpoint
def health_check():
    """Health check for load balancer."""
    try:
        # Basic system checks
        import psutil
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        if cpu_percent < 90 and memory_percent < 90:
            return {"status": "healthy", "cpu": cpu_percent, "memory": memory_percent}
        else:
            return {"status": "unhealthy", "cpu": cpu_percent, "memory": memory_percent}
    except Exception as e:
        return {"status": "error", "message": str(e)}
```

### 4. Load Balancing

```nginx
# nginx.conf
upstream multi_agent_rag {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
    server 127.0.0.1:8003;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://multi_agent_rag;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeout settings
        proxy_connect_timeout 30s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
    
    location /health {
        access_log off;
        proxy_pass http://multi_agent_rag/health;
    }
}
```

## Performance Optimization

### 1. Caching Strategy

```python
# caching.py
import redis
import json
import hashlib
from typing import Any, Optional

class CacheManager:
    """Redis-based caching for API responses."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
    
    def get_cache_key(self, message: str, mode: str) -> str:
        """Generate cache key for request."""
        content = f"{message}:{mode}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached response."""
        try:
            cached = self.redis_client.get(key)
            if cached:
                return json.loads(cached)
        except Exception:
            pass
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """Cache response."""
        try:
            self.redis_client.setex(key, ttl, json.dumps(value))
        except Exception:
            pass
```

### 2. Connection Pooling

```python
# connection_pool.py
import asyncio
from typing import Dict, List
from multi_agent_rag import MultiAgentRAGSystem

class SystemPool:
    """Pool of MultiAgentRAGSystem instances."""
    
    def __init__(self, pool_size: int = 4):
        self.pool_size = pool_size
        self.systems: List[MultiAgentRAGSystem] = []
        self.available: asyncio.Queue = asyncio.Queue()
        self.initialize_pool()
    
    def initialize_pool(self):
        """Initialize the system pool."""
        for _ in range(self.pool_size):
            system = MultiAgentRAGSystem(
                openai_key=os.environ['OPENAI_API_KEY'],
                tavily_key=os.environ['TAVILY_API_KEY']
            )
            self.systems.append(system)
            self.available.put_nowait(system)
    
    async def get_system(self) -> MultiAgentRAGSystem:
        """Get available system from pool."""
        return await self.available.get()
    
    async def return_system(self, system: MultiAgentRAGSystem):
        """Return system to pool."""
        await self.available.put(system)
```

## Troubleshooting

### Common Issues

#### 1. API Key Errors
```bash
# Check environment variables
echo $OPENAI_API_KEY
echo $TAVILY_API_KEY

# Verify API key format
python -c "import os; print('OpenAI:', len(os.environ.get('OPENAI_API_KEY', '')))"
```

#### 2. Memory Issues
```bash
# Monitor memory usage
top -p $(pgrep -f multi_agent_rag)

# Increase swap space
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 3. Network Issues
```bash
# Test API connectivity
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "test"}]}' \
     https://api.openai.com/v1/chat/completions
```

### Logging Configuration

```python
# logging_config.py
import logging
import sys

def setup_logging(level=logging.INFO):
    """Configure logging for production."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('multi_agent_rag.log')
        ]
    )
    
    # Suppress verbose library logging
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
```

## Maintenance

### 1. Regular Updates
```bash
# Update dependencies
pip install --upgrade -r requirements.txt

# Check for security vulnerabilities
pip audit

# Update system packages
sudo apt update && sudo apt upgrade
```

### 2. Backup Strategy
```bash
# Backup configuration
tar -czf backup_$(date +%Y%m%d).tar.gz .env config/

# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
backup_dir="/backups/multi_agent_rag_$DATE"
mkdir -p "$backup_dir"
cp -r . "$backup_dir/"
find /backups -name "multi_agent_rag_*" -mtime +7 -exec rm -rf {} \;
```

This deployment guide provides comprehensive instructions for deploying Jay Guwalani's Multi-Agent RAG System across various environments with production-ready configurations.
