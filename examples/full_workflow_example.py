#!/usr/bin/env python3
"""
Full Workflow Example - Jay Guwalani's Multi-Agent RAG System
Demonstrates complete research-to-document workflows
"""

import os
from dotenv import load_dotenv
from multi_agent_rag import MultiAgentRAGSystem

def main():
    """Demonstrate complete workflows using the multi-agent system."""
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
    
    # Workflow 1: Academic paper to blog post
    print("\n" + "="*80)
    print("WORKFLOW 1: Academic Paper Analysis to Technical Blog Post")
    print("="*80)
    
    workflow1_request = (
        "Research the paper 'Extending Llama-3's Context Ten-Fold Overnight' and "
        "create a comprehensive technical blog post explaining the methodology, "
        "results, and implications. Write it in a way that's accessible to ML engineers "
        "and save it as 'llama3_context_blog.md'."
    )
    
    print(f"Request: {workflow1_request}")
    print("\nProcessing complete workflow...")
    
    results1 = system.process_request(workflow1_request, recursion_limit=150)
    
    print("\nWorkflow Steps:")
    for i, step in enumerate(results1, 1):
        agent_name = list(step.keys())[0]
        if "messages" in step.get(agent_name, {}):
            message = step[agent_name]["messages"][0]
            print(f"\nStep {i} [{agent_name}]: {message.content[:300]}...")
    
    # Workflow 2: Market research to report
    print("\n" + "="*80)
    print("WORKFLOW 2: Market Research to Business Report")
    print("="*80)
    
    workflow2_request = (
        "Research the current state of AI in healthcare for 2024, including key trends, "
        "market size, major players, and challenges. Create a comprehensive business "
        "report with executive summary, market analysis, and recommendations. "
        "Save it as 'ai_healthcare_report_2024.md'."
    )
    
    print(f"Request: {workflow2_request}")
    print("\nProcessing complete workflow...")
    
    results2 = system.process_request(workflow2_request, recursion_limit=150)
    
    print("\nWorkflow Steps:")
    for i, step in enumerate(results2, 1):
        agent_name = list(step.keys())[0]
        if "messages" in step.get(agent_name, {}):
            message = step[agent_name]["messages"][0]
            print(f"\nStep {i} [{agent_name}]: {message.content[:300]}...")
    
    # Workflow 3: Technical documentation
    print("\n" + "="*80)
    print("WORKFLOW 3: Research to Technical Documentation")
    print("="*80)
    
    workflow3_request = (
        "Research best practices for implementing RAG systems in production environments. "
        "Include information about vector databases, chunking strategies, and performance "
        "optimization. Create a comprehensive technical guide for developers and "
        "save it as 'rag_production_guide.md'."
    )
    
    print(f"Request: {workflow3_request}")
    print("\nProcessing complete workflow...")
    
    results3 = system.process_request(workflow3_request, recursion_limit=150)
    
    print("\nWorkflow Steps:")
    for i, step in enumerate(results3, 1):
        agent_name = list(step.keys())[0]
        if "messages" in step.get(agent_name, {}):
            message = step[agent_name]["messages"][0]
            print(f"\nStep {i} [{agent_name}]: {message.content[:300]}...")
    
    # Workflow 4: Comparative analysis
    print("\n" + "="*80)
    print("WORKFLOW 4: Comparative Analysis to Decision Document")
    print("="*80)
    
    workflow4_request = (
        "Research and compare different multi-agent frameworks (LangGraph, CrewAI, AutoGen). "
        "Analyze their strengths, weaknesses, use cases, and performance characteristics. "
        "Create a detailed comparison document with recommendations for different scenarios. "
        "Save it as 'multiagent_framework_comparison.md'."
    )
    
    print(f"Request: {workflow4_request}")
    print("\nProcessing complete workflow...")
    
    results4 = system.process_request(workflow4_request, recursion_limit=150)
    
    print("\nWorkflow Steps:")
    for i, step in enumerate(results4, 1):
        agent_name = list(step.keys())[0]
        if "messages" in step.get(agent_name, {}):
            message = step[agent_name]["messages"][0]
            print(f"\nStep {i} [{agent_name}]: {message.content[:300]}...")
    
    print("\n" + "="*80)
    print("ALL WORKFLOWS COMPLETE!")
    print("="*80)
    
    # Show all created files
    if system.working_directory.exists():
        files = list(system.working_directory.rglob("*"))
        if files:
            print(f"\nAll files created in {system.working_directory}:")
            for file in files:
                print(f"  - {file.name}")
                try:
                    with open(file, 'r') as f:
                        content = f.read()
                        word_count = len(content.split())
                        print(f"    ({word_count} words)")
                except:
                    print("    (binary file)")
        else:
            print("\nNo files were created during the workflows.")
    
    print("\nWorkflows demonstrate the system's ability to:")
    print("- Conduct comprehensive research using multiple sources")
    print("- Synthesize information from academic papers and web sources")
    print("- Generate structured, professional documentation")
    print("- Coordinate between research and writing teams automatically")
    print("- Produce publication-ready content")

if __name__ == "__main__":
    main()
