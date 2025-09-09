#!/usr/bin/env python3
"""
Document Writing Example - Jay Guwalani's Multi-Agent RAG System
Demonstrates document creation and editing capabilities
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
    """Demonstrate document writing capabilities of the multi-agent system."""
    
    # Get API keys from environment or prompt user
    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    if not openai_key:
        openai_key = input("Enter OpenAI API Key: ").strip()
    
    if not tavily_key:
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
    
    try:
        results1 = system.document_only(outline_request)
        
        for step in results1:
            if isinstance(step, dict):
                for agent_name, content in step.items():
                    if agent_name != "supervisor" and isinstance(content, dict) and "messages" in content:
                        message = content["messages"][0]
                        msg_content = message.content if hasattr(message, 'content') else str(message)
                        print(f"\n[{agent_name}]: {msg_content}")
    except Exception as e:
        print(f"Error in example 1: {e}")
    
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
    
    try:
        results2 = system.document_only(document_request)
        
        for step in results2:
            if isinstance(step, dict):
                for agent_name, content in step.items():
                    if agent_name != "supervisor" and isinstance(content, dict) and "messages" in content:
                        message = content["messages"][0]
                        msg_content = message.content if hasattr(message, 'content') else str(message)
                        print(f"\n[{agent_name}]: {msg_content}")
    except Exception as e:
        print(f"Error in example 2: {e}")
    
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
    
    try:
        results3 = system.document_only(documentation_request)
        
        for step in results3:
            if isinstance(step, dict):
                for agent_name, content in step.items():
                    if agent_name != "supervisor" and isinstance(content, dict) and "messages" in content:
                        message = content["messages"][0]
                        msg_content = message.content if hasattr(message, 'content') else str(message)
                        print(f"\n[{agent_name}]: {msg_content}")
    except Exception as e:
        print(f"Error in example 3: {e}")
    
    print("\n" + "="*60)
    print("Document Writing Examples Complete!")
    print("="*60)
    
    # Show created files
    try:
        created_files = system.get_created_files()
        if created_files:
            print(f"\nFiles created in {system.working_directory}:")
            for file_path in created_files:
                relative_path = file_path.relative_to(system.working_directory)
                print(f"  - {relative_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        word_count = len(content.split())
                        print(f"    ({word_count} words)")
                except Exception as e:
                    print(f"    (Error reading file: {e})")
        else:
            print("\nNo files were created during the examples.")
    except Exception as e:
        print(f"Error listing files: {e}")

if __name__ == "__main__":
    main()
