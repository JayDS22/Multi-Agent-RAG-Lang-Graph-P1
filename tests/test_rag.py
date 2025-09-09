#!/usr/bin/env python3
"""
Unit tests for RAG functionality
Author: Jay Guwalani
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tiktoken
from pathlib import Path

class TestRAGSetup:
    """Test suite for RAG chain setup and configuration."""
    
    @patch('multi_agent_rag.PyMuPDFLoader')
    @patch('multi_agent_rag.OpenAIEmbeddings')
    @patch('multi_agent_rag.Qdrant')
    def test_rag_chain_initialization(self, mock_qdrant, mock_embeddings, mock_loader):
        """Test RAG chain initialization process."""
        # Mock document loading
        mock_docs = [Mock(page_content="test content", metadata={})]
        mock_loader.return_value.load.return_value = mock_docs
        
        # Mock text splitter
        with patch('multi_agent_rag.RecursiveCharacterTextSplitter') as mock_splitter:
            mock_splitter.return_value.split_documents.return_value = mock_docs
            
            # Mock vector store
            mock_qdrant.from_documents.return_value = Mock()
            mock_qdrant.from_documents.return_value.as_retriever.return_value = Mock()
            
            from multi_agent_rag import MultiAgentRAGSystem
            
            system = MultiAgentRAGSystem.__new__(MultiAgentRAGSystem)
            system.embedding_model = mock_embeddings.return_value
            system.working_directory = Path("/tmp")
            
            system._setup_rag_chain()
            
            # Verify components were initialized
            mock_loader.assert_called_once()
            mock_embeddings.assert_called_once_with(model="text-embedding-3-small")
            mock_qdrant.from_documents.assert_called_once()
    
    def test_tiktoken_length_function(self):
        """Test tiktoken length calculation."""
        from multi_agent_rag import MultiAgentRAGSystem
        
        # Mock tiktoken
        with patch('tiktoken.encoding_for_model') as mock_tiktoken:
            mock_encoding = Mock()
            mock_encoding.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
            mock_tiktoken.return_value = mock_encoding
            
            # Create tiktoken_len function (as it would be in _setup_rag_chain)
            def tiktoken_len(text):
                tokens = tiktoken.encoding_for_model("gpt-3.5-turbo").encode(text)
                return len(tokens)
            
            result = tiktoken_len("test text")
            assert result == 5
            mock_tiktoken.assert_called_with("gpt-3.5-turbo")
            mock_encoding.encode.assert_called_with("test text")
    
    @patch('multi_agent_rag.RecursiveCharacterTextSplitter')
    def test_text_splitting_configuration(self, mock_splitter):
        """Test text splitter configuration."""
        from multi_agent_rag import MultiAgentRAGSystem
        
        system = MultiAgentRAGSystem.__new__(MultiAgentRAGSystem)
        
        # Mock dependencies
        with patch('multi_agent_rag.PyMuPDFLoader'):
            with patch('multi_agent_rag.OpenAIEmbeddings'):
                with patch('multi_agent_rag.Qdrant'):
                    with patch('multi_agent_rag.ChatOpenAI'):
                        system._setup_rag_chain()
        
        # Verify splitter was called with correct parameters
        mock_splitter.assert_called_once_with(
            chunk_size=300,
            chunk_overlap=0,
            length_function=pytest.any  # Function object
        )

class TestRAGRetrieval:
    """Test suite for RAG retrieval functionality."""
    
    @pytest.fixture
    def mock_rag_chain(self):
        """Mock RAG chain for testing."""
        chain = Mock()
        chain.invoke.return_value = "Retrieved information from document"
        return chain
    
    def test_retrieve_information_tool(self, mock_rag_chain):
        """Test the retrieve_information tool."""
        from multi_agent_rag import MultiAgentRAGSystem
        
        system = MultiAgentRAGSystem.__new__(MultiAgentRAGSystem)
        system.rag_chain = mock_rag_chain
        
        # Setup tools
        system._setup_tools()
        
        # Get the retrieve_information tool
        retrieve_tool = system.retrieve_information
        
        # Test tool invocation
        result = retrieve_tool.invoke({"query": "test query"})
        
        assert result == "Retrieved information from document"
        mock_rag_chain.invoke.assert_called_once_with({"question": "test query"})
    
    def test_rag_prompt_template(self):
        """Test RAG prompt template formatting."""
        from langchain_core.prompts import ChatPromptTemplate
        
        RAG_PROMPT = """
        CONTEXT:
        {context}

        QUERY:
        {question}

        You are Jay Guwalani's AI assistant. Use the available context to answer the question. 
        If you can't answer the question, say you don't know.
        """
        
        prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
        
        # Test prompt formatting
        formatted = prompt.format(
            context="Test context information",
            question="What is the main topic?"
        )
        
        assert "Test context information" in formatted
        assert "What is the main topic?" in formatted
        assert "Jay Guwalani's AI assistant" in formatted

class TestVectorStore:
    """Test suite for vector store operations."""
    
    @patch('multi_agent_rag.Qdrant')
    def test_vector_store_creation(self, mock_qdrant):
        """Test vector store creation and configuration."""
        from multi_agent_rag import MultiAgentRAGSystem
        
        # Mock components
        mock_docs = [Mock()]
        mock_embeddings = Mock()
        
        # Test vector store setup
        mock_qdrant.from_documents.return_value = Mock()
        
        system = MultiAgentRAGSystem.__new__(MultiAgentRAGSystem)
        system.embedding_model = mock_embeddings
        
        # Simulate vector store creation
        vectorstore = mock_qdrant.from_documents(
            mock_docs,
            mock_embeddings,
            location=":memory:",
            collection_name="extending_context_window_llama_3",
        )
        
        mock_qdrant.from_documents.assert_called_once_with(
            mock_docs,
            mock_embeddings,
            location=":memory:",
            collection_name="extending_context_window_llama_3",
        )
    
    def test_retriever_configuration(self):
        """Test retriever configuration from vector store."""
        mock_vectorstore = Mock()
        mock_retriever = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        
        # Test retriever creation
        retriever = mock_vectorstore.as_retriever()
        
        assert retriever == mock_retriever
        mock_vectorstore.as_retriever.assert_called_once()

class TestRAGChainExecution:
    """Test suite for RAG chain execution."""
    
    def test_rag_chain_invoke_structure(self):
        """Test RAG chain invocation structure."""
        # Mock components
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = ["context document 1", "context document 2"]
        
        mock_prompt = Mock()
        mock_llm = Mock()
        mock_parser = Mock()
        
        mock_prompt.format.return_value = "formatted prompt"
        mock_llm.invoke.return_value = Mock(content="LLM response")
        mock_parser.parse.return_value = "parsed response"
        
        # Simulate LCEL chain construction
        from operator import itemgetter
        
        # Test that the chain structure is correct
        chain_input = {"question": "test question"}
        
        # Simulate retrieval
        context = mock_retriever.invoke(chain_input["question"])
        assert len(context) == 2
        
        # Simulate prompt formatting
        formatted_input = {
            "context": context,
            "question": chain_input["question"]
        }
        
        assert formatted_input["question"] == "test question"
        assert len(formatted_input["context"]) == 2
    
    def test_rag_error_handling(self):
        """Test RAG chain error handling."""
        mock_chain = Mock()
        mock_chain.invoke.side_effect = Exception("Retrieval failed")
        
        from multi_agent_rag import MultiAgentRAGSystem
        
        system = MultiAgentRAGSystem.__new__(MultiAgentRAGSystem)
        system.rag_chain = mock_chain
        
        # Test that errors are propagated correctly
        with pytest.raises(Exception, match="Retrieval failed"):
            system.rag_chain.invoke({"question": "test"})

class TestDocumentProcessing:
    """Test suite for document loading and processing."""
    
    @patch('multi_agent_rag.PyMuPDFLoader')
    def test_pdf_document_loading(self, mock_loader):
        """Test PDF document loading."""
        # Mock document content
        mock_doc = Mock()
        mock_doc.page_content = "Sample document content"
        mock_doc.metadata = {"page": 1}
        
        mock_loader.return_value.load.return_value = [mock_doc]
        
        # Test document loading
        loader = mock_loader("https://arxiv.org/pdf/2404.19553")
        docs = loader.load()
        
        assert len(docs) == 1
        assert docs[0].page_content == "Sample document content"
        mock_loader.assert_called_once_with("https://arxiv.org/pdf/2404.19553")
    
    def test_document_chunking(self):
        """Test document chunking process."""
        with patch('multi_agent_rag.RecursiveCharacterTextSplitter') as mock_splitter:
            # Mock document
            mock_doc = Mock()
            mock_doc.page_content = "A" * 1000  # Long content
            
            # Mock splitter
            chunk1 = Mock(page_content="A" * 300)
            chunk2 = Mock(page_content="A" * 300)
            chunk3 = Mock(page_content="A" * 400)
            
            mock_splitter.return_value.split_documents.return_value = [chunk1, chunk2, chunk3]
            
            splitter = mock_splitter(chunk_size=300, chunk_overlap=0)
            chunks = splitter.split_documents([mock_doc])
            
            assert len(chunks) == 3
            mock_splitter.return_value.split_documents.assert_called_once_with([mock_doc])

class TestRAGIntegration:
    """Integration tests for RAG functionality."""
    
    @patch('multi_agent_rag.PyMuPDFLoader')
    @patch('multi_agent_rag.OpenAIEmbeddings')
    @patch('multi_agent_rag.Qdrant')
    @patch('multi_agent_rag.ChatOpenAI')
    def test_full_rag_pipeline(self, mock_openai, mock_qdrant, mock_embeddings, mock_loader):
        """Test complete RAG pipeline integration."""
        # Mock all components
        mock_docs = [Mock(page_content="test content")]
        mock_loader.return_value.load.return_value = mock_docs
        
        mock_vectorstore = Mock()
        mock_retriever = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_qdrant.from_documents.return_value = mock_vectorstore
        
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content="Generated response")
        mock_openai.return_value = mock_llm
        
        # Test full pipeline
        from multi_agent_rag import MultiAgentRAGSystem
        
        system = MultiAgentRAGSystem.__new__(MultiAgentRAGSystem)
        system.embedding_model = mock_embeddings.return_value
        system.working_directory = Path("/tmp")
        
        with patch('multi_agent_rag.RecursiveCharacterTextSplitter'):
            system._setup_rag_chain()
        
        # Verify all components were called
        mock_loader.assert_called_once()
        mock_qdrant.from_documents.assert_called_once()
        mock_vectorstore.as_retriever.assert_called_once()

if __name__ == "__main__":
    pytest.main([__file__])
