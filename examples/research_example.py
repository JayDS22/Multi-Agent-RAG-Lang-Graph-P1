#!/usr/bin/env python3
"""
Research Example - Jay Guwalani's Multi-Agent RAG System
Demonstrates how to use the research team for information gathering
"""

import os
from dotenv import load_dotenv
from multi_agent_rag import MultiAgentRAGSystem

def main():
    """Demonstrate research capabilities of the multi-agent system."""
    # Load environment variables
    load_dotenv()
    
    # Get API keys from environment or prompt user
    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    if not openai_key or not tavily_key:
        print("Please set OPENAI_API_KEY and TAVILY_API_KEY in your .env file")
        return
    
    # Initialize the system
    print("Initializing Multi-Agent RAG System...")
    system = MultiAgentRAGSystem(openai_key, tavily_key)
    print("System initialized successfully!")
    
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
    
    results1 = system.research_only(research_query1)
    
    for step in results1:
        if "messages" in step.values().__iter__().__next__():
            agent_name = list(step.keys())[0]
            if agent_name != "supervisor":
                message = step[agent_name]["messages"][0]
                print(f"\n[{agent_name}]: {message.content[:500]}...")
    
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
    
    results2 = system.research_only(research_query2)
    
    for step in results2:
        if "messages" in step.values().__iter__().__next__():
            agent_name = list(step.keys())[0]
            if agent_name != "supervisor":
                message = step[agent_name]["messages"][0]
                print(f"\n[{agent_name}]: {message.content[:500]}...")
    
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
    
    results3 = system.research_only(research_query3)
    
    for step in results3:
        if "messages" in step.values().__iter__().__next__():
            agent_name = list(step.keys())[0]
            if agent_name != "supervisor":
                message = step[agent_name]["messages"][0]
                print(f"\n[{agent_name}]: {message.content[:500]}...")
    
    print("\n" + "="*60)
    print("Research Examples Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
