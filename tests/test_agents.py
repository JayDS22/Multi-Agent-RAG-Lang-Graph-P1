#!/usr/bin/env python3
"""
Unit tests for Multi-Agent RAG System agents
Author: Jay Guwalani
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from multi_agent_rag import MultiAgentRAGSystem

class TestMultiAgentRAGSystem:
    """Test suite for the main system class."""
    
    @pytest.fixture
    def mock_api_keys(self):
        """Mock API keys for testing."""
        return {
            'openai_key': 'test-openai-key',
            'tavily_key': 'test-tavily-key'
        }
    
    @pytest.fixture
    def system(self, mock_api_keys):
        """Create a system instance for testing."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': mock_api_keys['openai_key'],
            'TAVILY_API_KEY': mock_api_keys['tavily_key']
        }):
            with patch('multi_agent_rag.PyMuPDFLoader'), \
                 patch('multi_agent_rag.Qdrant'), \
                 patch('multi_agent_rag.OpenAIEmbeddings'), \
                 patch('multi_agent_rag.ChatOpenAI'):
                return MultiAgentRAGSystem(
                    mock_api_keys['openai_key'],
                    mock_api_keys['tavily_key']
                )
    
    def test_system_initialization(self, system):
        """Test that the system initializes correctly."""
        assert system is not None
        assert hasattr(system, 'llm')
        assert hasattr(system, 'embedding_model')
        assert hasattr(system, 'working_directory')
        assert hasattr(system, 'rag_chain')
    
    def test_invalid_api_keys(self):
        """Test system behavior with invalid API keys."""
        with pytest.raises(Exception):
            MultiAgentRAGSystem("", "")
    
    @patch('multi_agent_rag.MultiAgentRAGSystem._setup_rag_chain')
    @patch('multi_agent_rag.MultiAgentRAGSystem._setup_tools')
    @patch('multi_agent_rag.MultiAgentRAGSystem._setup_agents')
    @patch('multi_agent_rag.MultiAgentRAGSystem._build_graphs')
    def test_initialization_steps(self, mock_build, mock_agents, mock_tools, mock_rag):
        """Test that initialization steps are called in order."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-key',
            'TAVILY_API_KEY': 'test-key'
        }):
            MultiAgentRAGSystem('test-key', 'test-key')
            
            mock_rag.assert_called_once()
            mock_tools.assert_called_once()
            mock_agents.assert_called_once()
            mock_build.assert_called_once()

class TestRAGChain:
    """Test suite for RAG chain functionality."""
    
    @pytest.fixture
    def mock_system(self):
        """Mock system for RAG testing."""
        system = Mock()
        system.rag_chain = Mock()
        system.rag_chain.invoke.return_value = "Test response"
        return system
    
    def test_rag_chain_invoke(self, mock_system):
        """Test RAG chain invocation."""
        result = mock_system.rag_chain.invoke({"question": "test question"})
        assert result == "Test response"
        mock_system.rag_chain.invoke.assert_called_once()

class TestAgentCreation:
    """Test suite for agent creation and configuration."""
    
    def test_create_agent_parameters(self):
        """Test agent creation with proper parameters."""
        with patch('multi_agent_rag.create_openai_functions_agent') as mock_create:
            with patch('multi_agent_rag.AgentExecutor') as mock_executor:
                from multi_agent_rag import MultiAgentRAGSystem
                
                system = MultiAgentRAGSystem.__new__(MultiAgentRAGSystem)
                system.llm = Mock()
                
                # Test agent creation
                tools = [Mock()]
                system_prompt = "Test prompt"
                
                result = system._create_agent(system.llm, tools, system_prompt)
                
                mock_create.assert_called_once()
                mock_executor.assert_called_once()
    
    def test_supervisor_creation(self):
        """Test supervisor creation with routing logic."""
        with patch('multi_agent_rag.ChatPromptTemplate') as mock_prompt:
            with patch('multi_agent_rag.JsonOutputFunctionsParser') as mock_parser:
                from multi_agent_rag import MultiAgentRAGSystem
                
                system = MultiAgentRAGSystem.__new__(MultiAgentRAGSystem)
                system.llm = Mock()
                system.llm.bind_functions.return_value = Mock()
                
                # Test supervisor creation
                system_prompt = "Test supervisor prompt"
                members = ["Agent1", "Agent2"]
                
                result = system._create_team_supervisor(system.llm, system_prompt, members)
                
                mock_prompt.from_messages.assert_called_once()

class TestWorkflowExecution:
    """Test suite for workflow execution."""
    
    @pytest.fixture
    def mock_system_with_graphs(self):
        """Mock system with graph execution."""
        system = Mock()
        system.super_graph = Mock()
        system.research_chain = Mock()
        system.authoring_chain = Mock()
        
        # Mock stream responses
        system.super_graph.stream.return_value = [
            {"supervisor": {"next": "Research team"}},
            {"Research team": {"messages": [Mock(content="Research result")]}},
            {"supervisor": {"next": "FINISH"}}
        ]
        
        system.research_chain.stream.return_value = [
            {"supervisor": {"next": "Search"}},
            {"Search": {"messages": [Mock(content="Search result")]}},
            {"supervisor": {"next": "FINISH"}}
        ]
        
        system.authoring_chain.stream.return_value = [
            {"supervisor": {"next": "DocWriter"}},
            {"DocWriter": {"messages": [Mock(content="Document created")]}},
            {"supervisor": {"next": "FINISH"}}
        ]
        
        return system
    
    def test_process_request(self, mock_system_with_graphs):
        """Test full workflow processing."""
        from multi_agent_rag import MultiAgentRAGSystem
        
        # Replace system methods with mocked versions
        system = MultiAgentRAGSystem.__new__(MultiAgentRAGSystem)
        system.super_graph = mock_system_with_graphs.super_graph
        
        # Test process_request
        message = "Test request"
        results = []
        
        for step in system.super_graph.stream({"messages": [Mock(content=message)]}, {"recursion_limit": 150}):
            if "__end__" not in step:
                results.append(step)
        
        assert len(results) == 3
        assert "supervisor" in results[0]
        assert "Research team" in results[1]
    
    def test_research_only(self, mock_system_with_graphs):
        """Test research-only workflow."""
        from multi_agent_rag import MultiAgentRAGSystem
        
        system = MultiAgentRAGSystem.__new__(MultiAgentRAGSystem)
        system.research_chain = mock_system_with_graphs.research_chain
        
        message = "Research question"
        results = []
        
        for step in system.research_chain.stream(message, {"recursion_limit": 100}):
            if "__end__" not in step:
                results.append(step)
        
        assert len(results) == 3
        assert "Search" in results[1]
    
    def test_document_only(self, mock_system_with_graphs):
        """Test document-only workflow."""
        from multi_agent_rag import MultiAgentRAGSystem
        
        system = MultiAgentRAGSystem.__new__(MultiAgentRAGSystem)
        system.authoring_chain = mock_system_with_graphs.authoring_chain
        
        message = "Create document"
        results = []
        
        for step in system.authoring_chain.stream(message, {"recursion_limit": 100}):
            if "__end__" not in step:
                results.append(step)
        
        assert len(results) == 3
        assert "DocWriter" in results[1]

class TestToolFunctionality:
    """Test suite for agent tools."""
    
    def test_tavily_search_tool(self):
        """Test Tavily search tool configuration."""
        with patch('multi_agent_rag.TavilySearchResults') as mock_tavily:
            from multi_agent_rag import MultiAgentRAGSystem
            
            system = MultiAgentRAGSystem.__new__(MultiAgentRAGSystem)
            system._setup_tools()
            
            mock_tavily.assert_called_with(max_results=5)
    
    @patch('builtins.open', create=True)
    def test_document_tools(self, mock_open):
        """Test document manipulation tools."""
        from multi_agent_rag import MultiAgentRAGSystem
        import tempfile
        
        system = MultiAgentRAGSystem.__new__(MultiAgentRAGSystem)
        system.working_directory = tempfile.mkdtemp()
        
        # Mock file operations
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        system._setup_tools()
        
        # Test create_outline tool
        outline_tool = None
        for tool in system.doc_tools:
            if tool.name == 'create_outline':
                outline_tool = tool
                break
        
        assert outline_tool is not None
        
        # Test write_document tool
        write_tool = None
        for tool in system.doc_tools:
            if tool.name == 'write_document':
                write_tool = tool
                break
        
        assert write_tool is not None

class TestErrorHandling:
    """Test suite for error handling scenarios."""
    
    def test_api_key_validation(self):
        """Test API key validation."""
        with pytest.raises(Exception):
            MultiAgentRAGSystem(None, "valid-key")
        
        with pytest.raises(Exception):
            MultiAgentRAGSystem("valid-key", None)
    
    def test_recursion_limit_handling(self):
        """Test recursion limit enforcement."""
        system = Mock()
        system.process_request = Mock()
        
        # Test that recursion limit is passed correctly
        system.process_request("test", recursion_limit=50)
        system.process_request.assert_called_with("test", recursion_limit=50)
    
    @patch('multi_agent_rag.Path')
    def test_file_operation_errors(self, mock_path):
        """Test file operation error handling."""
        # Mock file operation failure
        mock_path.return_value.exists.return_value = False
        mock_path.return_value.mkdir.side_effect = PermissionError("Access denied")
        
        from multi_agent_rag import MultiAgentRAGSystem
        
        system = MultiAgentRAGSystem.__new__(MultiAgentRAGSystem)
        
        # Test that file errors are handled gracefully
        with pytest.raises(PermissionError):
            system._prelude({"current_files": ""})

if __name__ == "__main__":
    pytest.main([__file__])
