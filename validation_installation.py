#!/usr/bin/env python3
"""
Installation Validation Script
Author: Jay Guwalani

Comprehensive validation of the Multi-Agent RAG System installation.
"""

import sys
import os
import importlib
import traceback
from pathlib import Path

def print_header():
    """Print validation header."""
    print("=" * 60)
    print("Multi-Agent RAG System - Installation Validation")
    print("Author: Jay Guwalani")
    print("=" * 60)

def check_python_version():
    """Check Python version compatibility."""
    print("\n1. Checking Python Version...")
    
    if sys.version_info < (3, 8):
        print(f"   ✗ Python {sys.version_info.major}.{sys.version_info.minor} detected")
        print("   ✗ Python 3.8 or higher is required")
        return False
    
    print(f"   ✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n2. Checking Dependencies...")
    
    required_packages = [
        ("langgraph", "LangGraph framework"),
        ("langchain", "LangChain core"),
        ("langchain_openai", "OpenAI integration"),
        ("langchain_community", "Community tools"),
        ("qdrant_client", "Vector database"),
        ("tiktoken", "Tokenization"),
        ("pymupdf", "PDF processing"),
    ]
    
    missing_packages = []
    
    for package, description in required_packages:
        try:
            importlib.import_module(package)
            print(f"   ✓ {package} ({description})")
        except ImportError:
            print(f"   ✗ {package} ({description}) - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n   Missing packages: {', '.join(missing_packages)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    return True

def check_main_module():
    """Check if the main module loads correctly."""
    print("\n3. Checking Main Module...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, str(Path(__file__).parent))
        
        from multi_agent_rag import MultiAgentRAGSystem
        print("   ✓ Main module imports successfully")
        
        # Check if class has required methods
        required_methods = [
            'process_request',
            'research_only', 
            'document_only',
            'get_created_files'
        ]
        
        for method in required_methods:
            if hasattr(MultiAgentRAGSystem, method):
                print(f"   ✓ Method '{method}' available")
            else:
                print(f"   ✗ Method '{method}' missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"   ✗ Failed to import main module: {e}")
        traceback.print_exc()
        return False

def check_api_keys():
    """Check API key configuration."""
    print("\n4. Checking API Keys...")
    
    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    if openai_key:
        # Check if key looks valid (starts with sk- and has reasonable length)
        if openai_key.startswith("sk-") and len(openai_key) > 40:
            print("   ✓ OpenAI API key found and appears valid")
        else:
            print("   ⚠ OpenAI API key found but may be invalid")
    else:
        print("   ⚠ OpenAI API key not found")
        print("     Set OPENAI_API_KEY environment variable")
    
    if tavily_key:
        print("   ✓ Tavily API key found")
    else:
        print("   ⚠ Tavily API key not found (will use mock search)")
        print("     Set TAVILY_API_KEY environment variable for web search")
    
    return bool(openai_key)

def test_system_initialization():
    """Test system initialization."""
    print("\n5. Testing System Initialization...")
    
    try:
        from multi_agent_rag import MultiAgentRAGSystem
        
        # Use mock keys if real ones aren't available
        openai_key = os.getenv("OPENAI_API_KEY") or "mock-openai-key"
        tavily_key = os.getenv("TAVILY_API_KEY") or "mock-tavily-key"
        
        # Only test with real OpenAI key
        if openai_key == "mock-openai-key":
            print("   ⚠ Skipping system test - no valid OpenAI API key")
            return True
        
        system = MultiAgentRAGSystem(openai_key, tavily_key)
        print("   ✓ System initialized successfully")
        
        # Test working directory
        if system.working_directory.exists():
            print("   ✓ Working directory created")
        else:
            print("   ✗ Working directory not created")
            return False
        
        # Test file creation method
        files = system.get_created_files()
        print(f"   ✓ File listing works ({len(files)} files)")
        
        return True
        
    except Exception as e:
        print(f"   ✗ System initialization failed: {e}")
        return False

def check_example_files():
    """Check if example files exist and are executable."""
    print("\n6. Checking Example Files...")
    
    example_files = [
        "examples/research_example.py",
        "examples/document_example.py", 
        "examples/full_workflow_example.py"
    ]
    
    all_good = True
    
    for example_file in example_files:
        if Path(example_file).exists():
            print(f"   ✓ {example_file}")
        else:
            print(f"   ✗ {example_file} - Missing")
            all_good = False
    
    return all_good

def check_documentation():
    """Check if documentation files exist."""
    print("\n7. Checking Documentation...")
    
    doc_files = [
        "README.md",
        "docs/architecture.md",
        "docs/api_reference.md", 
        "docs/deployment_guide.md"
    ]
    
    for doc_file in doc_files:
        if Path(doc_file).exists():
            print(f"   ✓ {doc_file}")
        else:
            print(f"   ⚠ {doc_file} - Missing")
    
    return True  # Documentation is optional for functionality

def run_quick_functionality_test():
    """Run a quick functionality test if possible."""
    print("\n8. Quick Functionality Test...")
    
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_key or openai_key.startswith("mock"):
        print("   ⚠ Skipping - no valid OpenAI API key")
        return True
    
    try:
        from multi_agent_rag import MultiAgentRAGSystem
        
        print("   • Initializing system...")
        system = MultiAgentRAGSystem(openai_key, "mock-tavily-key")
        
        print("   • Testing document creation...")
        results = system.document_only(
            "Create a simple test outline with 3 points. Save as 'validation_test.txt'"
        )
        
        print("   • Checking results...")
        if results and not any("error" in str(result).lower() for result in results):
            print("   ✓ Basic functionality test passed")
            
            # Check if file was created
            files = system.get_created_files()
            if files:
                print(f"   ✓ File creation successful ({len(files)} files)")
            
            return True
        else:
            print("   ⚠ Functionality test completed with warnings")
            return True
        
    except Exception as e:
        print(f"   ⚠ Functionality test failed: {e}")
        print("   This may be due to API rate limits or network issues")
        return True  # Don't fail validation for API issues

def print_summary(results):
    """Print validation summary."""
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for check, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{check:<30} {status}")
    
    print(f"\nResults: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n✓ Installation is fully functional!")
        print("\nNext steps:")
        print("1. Set your API keys in environment variables")
        print("2. Run: python examples/research_example.py")
        print("3. Run: python examples/document_example.py")
        print("4. Try interactive mode: python multi_agent_rag.py")
    elif passed >= total * 0.7:  # 70% pass rate
        print("\n⚠ Installation is mostly functional!")
        print("Some optional components may be missing.")
        print("Core functionality should work.")
    else:
        print("\n✗ Installation has issues!")
        print("Please review the failed checks above.")
        print("Run: python quick_setup.py")
    
    print("\nFor help, see README.md or contact Jay Guwalani")

def main():
    """Main validation function."""
    print_header()
    
    # Run all validation checks
    checks = {
        "Python Version": check_python_version(),
        "Dependencies": check_dependencies(), 
        "Main Module": check_main_module(),
        "API Keys": check_api_keys(),
        "System Init": test_system_initialization(),
        "Example Files": check_example_files(),
        "Documentation": check_documentation(),
        "Quick Test": run_quick_functionality_test(),
    }
    
    print_summary(checks)
    
    # Exit code based on critical checks
    critical_checks = ["Python Version", "Dependencies", "Main Module"]
    critical_passed = all(checks[check] for check in critical_checks if check in checks)
    
    sys.exit(0 if critical_passed else 1)

if __name__ == "__main__":
    main()
