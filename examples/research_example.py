#!/usr/bin/env python3
"""
Research Example - Jay Guwalani's Multi-Agent RAG System
Demonstrates how to use the research team for information gathering
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Note: python-dotenv not installed. Using environment variables directly.")

from multi_agent_rag import MultiAgentRAGSystem

def main():
    """Demonstrate research capabilities of the multi-agent system."""
    
    # Get API keys from environment or prompt user
    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    if not openai_key:
        print("Please set OPENAI_API_KEY environment variable or in .env file")
        openai_key = input("Enter OpenAI API Key: ").strip()
    
    if not tavily_key:
        print("Please set TAVILY_API_KEY environment variable or in .env file")
        tavily_key = input("Enter Tavily API Key: ").strip()
    
    if not openai_key or not tavily_key:
        print("Both API keys are required!")
        return
    
    # Initialize the system
    print("Initializing Multi-Agent RAG System...")
    try:
        system = MultiAgentRAGSystem(openai_key, tavily_key)
        print("System initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize system: {e}")
        return
    
    # Example 1: Academic paper research
    print("\n" + "="*60)
    print("Example 1: Academic Paper Analysis")
    print("="*60)
    
    research_query1 = (
        "What are the main technical contributions of the paper "
        "'Extending Llama-3's Context Ten-Fold Overnight'? "
        "Focus on the methodology and results."
    )
    
    print(f"Query: {research_query1}")
    print("\nProcessing...")
    
    try:
        results1 = system.research_only(research_query1)
        
        for step in results1:
            if isinstance(step, dict) and any(isinstance(v, dict) and "messages" in v for v in step.values()):
                agent_name = list(step.keys())[0]
                if agent_name != "supervisor" and "messages" in step[agent_name]:
                    message = step[agent_name]["messages"][0]
                    content = message.content if hasattr(message, 'content') else str(message)
                    print(f"\n[{agent_name}]: {content[:500]}...")
    except Exception as e:
        print(f"Error in research query 1: {e}")
    
    # Example 2: Current technology trends
    print("\n" + "="*60)
    print("Example 2: Current Technology Research")
    print("="*60)
    
    research_query2 = (
        "What are the latest developments in large language model "
        "context windows in 2024? Include recent research papers and industry developments."
    )
    
    print(f"Query: {research_query2}")
    print("\nProcessing...")
    
    try:
        results2 = system.research_only(research_query2)
        
        for step in results2:
            if isinstance(step, dict) and any(isinstance(v, dict) and "messages" in v for v in step.values()):
                agent_name = list(step.keys())[0]
                if agent_name != "supervisor" and "messages" in step[agent_name]:
                    message = step[agent_name]["messages"][0]
                    content = message.content if hasattr(message, 'content') else str(message)
                    print(f"\n[{agent_name}]: {content[:500]}...")
    except Exception as e:
        print(f"Error in research query 2: {e}")
    
    # Example 3: Comparative analysis
    print("\n" + "="*60)
    print("Example 3: Comparative Research")
    print("="*60)
    
    research_query3 = (
        "Compare the QLoRA fine-tuning approach mentioned in the Llama-3 context paper "
        "with other parameter-efficient fine-tuning methods. What are the advantages?"
    )
    
    print(f"Query: {research_query3}")
    print("\nProcessing...")
    
    try:
        results3 = system.research_only(research_query3)
        
        for step in results3:
            if isinstance(step, dict) and any(isinstance(v, dict) and "messages" in v for v in step.values()):
                agent_name = list(step.keys())[0]
                if agent_name != "supervisor" and "messages" in step[agent_name]:
                    message = step[agent_name]["messages"][0]
                    content = message.content if hasattr(message, 'content') else str(message)
                    print(f"\n[{agent_name}]: {content[:500]}...")
    except Exception as e:
        print(f"Error in research query 3: {e}")
    
    print("\n" + "="*60)
    print("Research Examples Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
