#!/usr/bin/env python3
"""
Document Writing Example - Jay Guwalani's Multi-Agent RAG System
Demonstrates document creation and editing capabilities
"""

import os
from dotenv import load_dotenv
from multi_agent_rag import MultiAgentRAGSystem

def main():
    """Demonstrate document writing capabilities of the multi-agent system."""
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
    
    # Example 1: Technical blog outline
    print("\n" + "="*60)
    print("Example 1: Creating a Technical Blog Outline")
    print("="*60)
    
    outline_request = (
        "Create a detailed outline for a technical blog post about "
        "implementing RAG systems in production. Include sections on "
        "architecture, challenges, and best practices."
    )
    
    print(f"Request: {outline_request}")
    print("\nProcessing...")
    
    results1 = system.document_only(outline_request)
    
    for step in results1:
        if "messages" in step.values().__iter__().__next__():
            agent_name = list(step.keys())[0]
            if agent_name != "supervisor":
                message = step[agent_name]["messages"][0]
                print(f"\n[{agent_name}]: {message.content}")
    
    # Example 2: Complete document creation
    print("\n" + "="*60)
    print("Example 2: Writing a Complete Technical Document")
    print("="*60)
    
    document_request = (
        "Write a comprehensive technical guide on 'Multi-Agent Systems in AI'. "
        "Include an introduction, key concepts, implementation strategies, "
        "and conclusion. Save it as 'multi_agent_guide.md'."
    )
    
    print(f"Request: {document_request}")
    print("\nProcessing...")
    
    results2 = system.document_only(document_request)
    
    for step in results2:
        if "messages" in step.values().__iter__().__next__():
            agent_name = list(step.keys())[0]
            if agent_name != "supervisor":
                message = step[agent_name]["messages"][0]
                print(f"\n[{agent_name}]: {message.content}")
    
    # Example 3: Documentation structure
    print("\n" + "="*60)
    print("Example 3: Creating Project Documentation")
    print("="*60)
    
    documentation_request = (
        "Create a structured README outline for a machine learning project. "
        "Include sections for installation, usage, API reference, and examples. "
        "Make it suitable for a GitHub repository."
    )
    
    print(f"Request: {documentation_request}")
    print("\nProcessing...")
    
    results3 = system.document_only(documentation_request)
    
    for step in results3:
        if "messages" in step.values().__iter__().__next__():
            agent_name = list(step.keys())[0]
            if agent_name != "supervisor":
                message = step[agent_name]["messages"][0]
                print(f"\n[{agent_name}]: {message.content}")
    
    # Example 4: Meeting notes and summaries
    print("\n" + "="*60)
    print("Example 4: Creating Meeting Notes Template")
    print("="*60)
    
    notes_request = (
        "Create a template for technical team meeting notes. "
        "Include sections for action items, decisions, and follow-ups. "
        "Save it as 'meeting_notes_template.md'."
    )
    
    print(f"Request: {notes_request}")
    print("\nProcessing...")
    
    results4 = system.document_only(notes_request)
    
    for step in results4:
        if "messages" in step.values().__iter__().__next__():
            agent_name = list(step.keys())[0]
            if agent_name != "supervisor":
                message = step[agent_name]["messages"][0]
                print(f"\n[{agent_name}]: {message.content}")
    
    print("\n" + "="*60)
    print("Document Writing Examples Complete!")
    print("="*60)
    
    # Show created files
    if system.working_directory.exists():
        files = list(system.working_directory.rglob("*"))
        if files:
            print(f"\nFiles created in {system.working_directory}:")
            for file in files:
                print(f"  - {file.name}")

if __name__ == "__main__":
    main()
