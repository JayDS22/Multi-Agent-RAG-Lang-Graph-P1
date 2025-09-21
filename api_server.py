#!/usr/bin/env python3
"""
FastAPI Server for Production Multi-Agent RAG
Author: Jay Guwalani

Provides REST API for load testing and integration
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
from production_multiagent_rag import ProductionMultiAgentRAG
import uvicorn

# Initialize FastAPI
app = FastAPI(
    title="Multi-Agent RAG API",
    description="Production-ready multi-agent RAG system with performance monitoring",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class ProcessRequest(BaseModel):
    message: str
    mode: str = "full"  # full, research, document

class HealthResponse(BaseModel):
    status: str
    version: str
    monitoring: Dict[str, bool]

# Global system instance
system: Optional[ProductionMultiAgentRAG] = None

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    global system
    
    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    if not openai_key:
        raise ValueError("OPENAI_API_KEY environment variable required")
    
    system = ProductionMultiAgentRAG(
        openai_key=openai_key,
        tavily_key=tavily_key,
        enable_monitoring=True
    )
    
    print("✅ Multi-Agent RAG API Server initialized")

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if system else "initializing",
        "version": "1.0.0",
        "monitoring": {
            "system": system.system_monitor.get_current_stats()["available"] if system else False,
            "gpu": system.gpu_monitor.get_current_stats()["available"] if system else False
        }
    }

@app.post("/process")
async def process_request(request: ProcessRequest):
    """Process a request through the multi-agent system"""
    if not system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        results = system.process(request.message)
        metrics = system.get_metrics()
        
        return {
            "success": True,
            "results": results,
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/research")
async def research_only(request: ProcessRequest):
    """Research-only endpoint"""
    if not system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        results = []
        for step in system.research_chain.stream(request.message):
            if "__end__" not in step:
                results.append(step)
        
        return {
            "success": True,
            "results": results,
            "metrics": system.get_metrics()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/document")
async def document_only(request: ProcessRequest):
    """Document creation endpoint"""
    if not system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        results = []
        for step in system.doc_chain.stream(request.message):
            if "__end__" not in step:
                results.append(step)
        
        return {
            "success": True,
            "results": results,
            "metrics": system.get_metrics()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get current system metrics"""
    if not system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return system.get_metrics()

@app.post("/cache/clear")
async def clear_cache():
    """Clear the semantic cache"""
    if not system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    system.cache.cache.clear()
    return {"success": True, "message": "Cache cleared"}

@app.get("/files")
async def list_files():
    """List created files"""
    if not system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        files = []
        if system.working_directory.exists():
            files = [
                {
                    "name": f.name,
                    "size": f.stat().st_size,
                    "modified": f.stat().st_mtime
                }
                for f in system.working_directory.rglob("*") if f.is_file()
            ]
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
