#!/usr/bin/env python3
"""
Quick Setup Script for Multi-Agent RAG System
Author: Jay Guwalani

This script helps verify the installation and run basic tests.
"""

import sys
import subprocess
import os
from pathlib import Path

def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing dependencies: {e}")
        return False

def check_api_keys():
    """Check if API keys are available."""
    print("Checking API keys...")
    
    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    if not openai_key:
        print("⚠ OPENAI_API_KEY not found in environment")
        openai_key = input("Enter OpenAI API Key (or press Enter to skip): ").strip()
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
    
    if not tavily_key:
        print("⚠ TAVILY_API_KEY not found in environment")
        print("Note: You can get a Tavily API key from https://tavily.com/")
        tavily_key = input("Enter Tavily API Key (or press Enter to use mock): ").strip()
        if tavily_key:
            os.environ["TAVILY_API_KEY"] = tavily_key
        else:
            os.environ["TAVILY_API_KEY"] = "mock-key"  # Will use mock search
    
    return bool(os.getenv("OPENAI_API_KEY"))

def test_basic_functionality():
    """Test basic system functionality."""
    print("Testing basic functionality...")
    
    try:
        from multi_agent_rag import MultiAgentRAGSystem
        
        openai_key = os.getenv("OPENAI_API_KEY")
        tavily_key = os.getenv("TAVILY_API_KEY", "mock-key")
        
        if not openai_key:
            print("✗ Cannot test without OpenAI API key")
            return False
        
        # Initialize system
        system = MultiAgentRAGSystem(openai_key, tavily_key)
        print("✓ System initialized successfully")
        
        # Test file creation
        test_files = system.get_created_files()
        print(f"✓ Working directory created: {system.working_directory}")
        
        print("✓ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def run_simple_example():
    """Run a simple example to verify everything works."""
    print("Running simple example...")
    
    try:
        from multi_agent_rag import MultiAgentRAGSystem
        
        openai_key = os.getenv("OPENAI_API_KEY")
        tavily_key = os.getenv("TAVILY_API_KEY", "mock-key")
        
        if not openai_key:
            print("⚠ Skipping example - no OpenAI API key")
            return True
        
        system = MultiAgentRAGSystem(openai_key, tavily_key)
        
        # Simple document creation test
        print("Testing document creation...")
        results = system.document_only(
            "Create a simple outline for a technical blog about AI. Save it as 'test_outline.txt'."
        )
        
        # Check if any files were created
        files = system.get_created_files()
        if files:
            print(f"✓ Created {len(files)} file(s)")
            for file_path in files:
                print(f"  - {file_path.name}")
        else:
            print("⚠ No files were created, but system ran without errors")
        
        print("✓ Simple example completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Simple example failed: {e}")
        return False

def main():
    """Main setup function."""
    print("Multi-Agent RAG System - Quick Setup")
    print("=" * 50)
    print("Author: Jay Guwalani")
    print()
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Install dependencies
    if not install_dependencies():
        print("Setup failed at dependency installation")
        sys.exit(1)
    
    # Check API keys
    has_openai = check_api_keys()
    
    # Test basic functionality
    if not test_basic_functionality():
        print("Setup failed at basic functionality test")
        sys.exit(1)
    
    # Run simple example if we have API keys
    if has_openai:
        if not run_simple_example():
            print("Setup completed but example failed")
        else:
            print("\n" + "=" * 50)
            print("✓ Setup completed successfully!")
            print("\nNext steps:")
            print("1. Run: python examples/research_example.py")
            print("2. Run: python examples/document_example.py")
            print("3. Run: python examples/full_workflow_example.py")
            print("4. Or use interactively: python multi_agent_rag.py")
    else:
        print("\n" + "=" * 50)
        print("⚠ Setup completed with limitations")
        print("Note: Some features require valid API keys")
        print("Set OPENAI_API_KEY and TAVILY_API_KEY environment variables")
    
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()
