# Multi-Agent RAG System - Jay Guwalani
# A hierarchical multi-agent system combining RAG with document generation capabilities

import os
import functools
import operator
import tiktoken
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, List, Optional, TypedDict, Union, Dict, Annotated

# Core LangChain and LangGraph imports
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

from langgraph.graph import END, StateGraph


class MultiAgentRAGSystem:
    """
    A comprehensive multi-agent system built by Jay Guwalani for handling
    research, document analysis, and content generation tasks.
    """
    
    def __init__(self, openai_key: str, tavily_key: str):
        """Initialize the multi-agent system with API keys."""
        if not openai_key or not tavily_key:
            raise ValueError("Both OpenAI and Tavily API keys are required")
            
        os.environ["OPENAI_API_KEY"] = openai_key
        os.environ["TAVILY_API_KEY"] = tavily_key
        
        # Initialize core components
        self.llm = ChatOpenAI(model="gpt-4-1106-preview")
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Create working directory
        self._temp_dir = TemporaryDirectory()
        self.working_directory = Path(self._temp_dir.name)
        self.working_directory.mkdir(exist_ok=True)
        
        # Initialize components in correct order
        self._setup_rag_chain()
        self._setup_tools()
        self._setup_agents()
        self._build_graphs()
        
    def __del__(self):
        """Cleanup temporary directory."""
        if hasattr(self, '_temp_dir'):
            self._temp_dir.cleanup()
        
    def _setup_rag_chain(self):
        """Set up the RAG chain for document processing."""
        try:
            # Load and process document
            docs = PyMuPDFLoader("https://arxiv.org/pdf/2404.19553").load()
            
            # Text splitting with tiktoken
            def tiktoken_len(text):
                tokens = tiktoken.encoding_for_model("gpt-3.5-turbo").encode(text)
                return len(tokens)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,
                chunk_overlap=0,
                length_function=tiktoken_len,
            )
            
            split_chunks = text_splitter.split_documents(docs)
            
            # Create vector store
            self.qdrant_vectorstore = Qdrant.from_documents(
                split_chunks,
                self.embedding_model,
                location=":memory:",
                collection_name="extending_context_window_llama_3",
            )
            
            self.qdrant_retriever = self.qdrant_vectorstore.as_retriever()
            
            # RAG prompt template
            RAG_PROMPT = """
            CONTEXT:
            {context}

            QUERY:
            {question}

            You are Jay Guwalani's AI assistant. Use the available context to answer the question. 
            If you can't answer the question, say you don't know.
            """
            
            rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
            openai_chat_model = ChatOpenAI(model="gpt-3.5-turbo")
            
            # Build RAG chain
            self.rag_chain = (
                {"context": itemgetter("question") | self.qdrant_retriever, 
                 "question": itemgetter("question")}
                | rag_prompt | openai_chat_model | StrOutputParser()
            )
        except Exception as e:
            print(f"Warning: Could not load RAG document: {e}")
            # Create a simple fallback RAG chain
            self.rag_chain = lambda x: "RAG system not available due to document loading error."
    
    def _setup_tools(self):
        """Set up all tools for the multi-agent system."""
        # Search tool
        try:
            self.tavily_tool = TavilySearchResults(max_results=5)
        except Exception as e:
            print(f"Warning: Tavily search not available: {e}")
            # Create mock search tool
            @tool
            def mock_search(query: str) -> str:
                """Mock search tool when Tavily is not available."""
                return f"Search results for '{query}' (mock response)"
            self.tavily_tool = mock_search
        
        # RAG tool
        @tool
        def retrieve_information(
            query: Annotated[str, "query to ask the retrieve information tool"]
        ):
            """Use Retrieval Augmented Generation to retrieve information about the 'Extending Llama-3's Context Ten-Fold Overnight' paper."""
            try:
                if callable(self.rag_chain):
                    return self.rag_chain({"question": query})
                else:
                    return self.rag_chain.invoke({"question": query})
            except Exception as e:
                return f"RAG retrieval error: {e}"
        
        self.retrieve_information = retrieve_information
        
        # Document tools
        @tool
        def create_outline(
            points: Annotated[List[str], "List of main points or sections."],
            file_name: Annotated[str, "File path to save the outline."],
        ) -> Annotated[str, "Path of the saved outline file."]:
            """Create and save an outline."""
            try:
                file_path = self.working_directory / file_name
                with file_path.open("w", encoding='utf-8') as file:
                    for i, point in enumerate(points):
                        file.write(f"{i + 1}. {point}\n")
                return f"Outline saved to {file_name}"
            except Exception as e:
                return f"Error creating outline: {e}"

        @tool
        def read_document(
            file_name: Annotated[str, "File path to save the document."],
            start: Annotated[Optional[int], "The start line. Default is 0"] = None,
            end: Annotated[Optional[int], "The end line. Default is None"] = None,
        ) -> str:
            """Read the specified document."""
            try:
                file_path = self.working_directory / file_name
                if not file_path.exists():
                    return f"File {file_name} does not exist."
                
                with file_path.open("r", encoding='utf-8') as file:
                    lines = file.readlines()
                
                if start is None:
                    start = 0
                return "".join(lines[start:end])
            except Exception as e:
                return f"Error reading document: {e}"

        @tool
        def write_document(
            content: Annotated[str, "Text content to be written into the document."],
            file_name: Annotated[str, "File path to save the document."],
        ) -> Annotated[str, "Path of the saved document file."]:
            """Create and save a text document."""
            try:
                file_path = self.working_directory / file_name
                with file_path.open("w", encoding='utf-8') as file:
                    file.write(content)
                return f"Document saved to {file_name}"
            except Exception as e:
                return f"Error writing document: {e}"

        @tool
        def edit_document(
            file_name: Annotated[str, "Path of the document to be edited."],
            inserts: Annotated[
                Dict[int, str],
                "Dictionary where key is the line number (1-indexed) and value is the text to be inserted at that line.",
            ],
        ) -> Annotated[str, "Path of the edited document file."]:
            """Edit a document by inserting text at specific line numbers."""
            try:
                file_path = self.working_directory / file_name
                if not file_path.exists():
                    return f"File {file_name} does not exist."
                
                with file_path.open("r", encoding='utf-8') as file:
                    lines = file.readlines()

                sorted_inserts = sorted(inserts.items())

                for line_number, text in sorted_inserts:
                    if 1 <= line_number <= len(lines) + 1:
                        lines.insert(line_number - 1, text + "\n")
                    else:
                        return f"Error: Line number {line_number} is out of range."

                with file_path.open("w", encoding='utf-8') as file:
                    file.writelines(lines)

                return f"Document edited and saved to {file_name}"
            except Exception as e:
                return f"Error editing document: {e}"
        
        self.doc_tools = [create_outline, read_document, write_document, edit_document]
    
    def _create_agent(self, llm: ChatOpenAI, tools: list, system_prompt: str) -> AgentExecutor:
        """Create a function-calling agent."""
        enhanced_prompt = system_prompt + (
            "\nWork autonomously according to your specialty, using the tools available to you."
            " Do not ask for clarification."
            " Your other team members (and other teams) will collaborate with you with their own specialties."
            " You are chosen for a reason!"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", enhanced_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_functions_agent(llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools)
        return executor

    def _create_team_supervisor(self, llm: ChatOpenAI, system_prompt: str, members: List[str]) -> Runnable:
        """An LLM-based router."""
        options = ["FINISH"] + members
        function_def = {
            "name": "route",
            "description": "Select the next role.",
            "parameters": {
                "title": "routeSchema",
                "type": "object",
                "properties": {
                    "next": {
                        "title": "Next",
                        "anyOf": [{"enum": options}],
                    },
                },
                "required": ["next"],
            },
        }
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            ("system", "Given the conversation above, who should act next?"
             " Or should we FINISH? Select one of: {options}"),
        ]).partial(options=str(options), team_members=", ".join(members))
        
        return (
            prompt
            | llm.bind_functions(functions=[function_def], function_call="route")
            | JsonOutputFunctionsParser()
        )

    def _agent_node(self, state, agent, name):
        """Helper function to create agent nodes."""
        try:
            result = agent.invoke(state)
            return {"messages": [HumanMessage(content=result["output"], name=name)]}
        except Exception as e:
            error_msg = f"Agent {name} encountered an error: {str(e)}"
            return {"messages": [HumanMessage(content=error_msg, name=name)]}

    def _setup_agents(self):
        """Set up all agents for the system."""
        # Research team agents
        self.search_agent = self._create_agent(
            self.llm,
            [self.tavily_tool],
            "You are Jay Guwalani's research assistant who can search for up-to-date info using the search engine.",
        )
        
        self.research_agent = self._create_agent(
            self.llm,
            [self.retrieve_information],
            "You are Jay Guwalani's research assistant who can provide specific information on the provided paper: 'Extending Llama-3's Context Ten-Fold Overnight'",
        )
        
        # Document team agents
        self.doc_writer_agent = self._create_agent(
            self.llm,
            [self.doc_tools[2], self.doc_tools[3], self.doc_tools[1]],  # write, edit, read
            "You are Jay Guwalani's expert writing assistant for research documents.",
        )
        
        self.note_taking_agent = self._create_agent(
            self.llm,
            [self.doc_tools[0], self.doc_tools[1]],  # create_outline, read_document
            "You are Jay Guwalani's expert senior researcher tasked with writing blog outlines and taking notes to craft perfect technical content.",
        )
        
        # Supervisors
        self.research_supervisor = self._create_team_supervisor(
            self.llm,
            "You are a supervisor tasked with managing a conversation between the"
            " following workers: Search, PaperInformationRetriever. Given the following user request,"
            " respond with the worker to act next. Each worker will perform a"
            " task and respond with their results and status. When finished,"
            " respond with FINISH.",
            ["Search", "PaperInformationRetriever"],
        )
        
        self.doc_supervisor = self._create_team_supervisor(
            self.llm,
            "You are a supervisor tasked with managing a conversation between the"
            " following workers: DocWriter, NoteTaker. Given the following user request,"
            " respond with the worker to act next. Each worker will perform a"
            " task and respond with their results and status. When finished,"
            " respond with FINISH.",
            ["DocWriter", "NoteTaker"],
        )

    def _prelude(self, state):
        """Helper function to track current files."""
        written_files = []
        try:
            if self.working_directory.exists():
                written_files = [
                    f.relative_to(self.working_directory) 
                    for f in self.working_directory.rglob("*") if f.is_file()
                ]
        except Exception:
            pass
            
        if not written_files:
            return {**state, "current_files": "No files written."}
        return {
            **state,
            "current_files": "\nBelow are files your team has written to the directory:\n"
            + "\n".join([f" - {f}" for f in written_files]),
        }

    def _build_graphs(self):
        """Build the hierarchical graph structure."""
        # Define states
        class ResearchTeamState(TypedDict):
            messages: Annotated[List[BaseMessage], operator.add]
            team_members: List[str]
            next: str

        class DocWritingState(TypedDict):
            messages: Annotated[List[BaseMessage], operator.add]
            team_members: str
            next: str
            current_files: str

        class State(TypedDict):
            messages: Annotated[List[BaseMessage], operator.add]
            next: str

        # Create agent nodes
        search_node = functools.partial(self._agent_node, agent=self.search_agent, name="Search")
        research_node = functools.partial(self._agent_node, agent=self.research_agent, name="PaperInformationRetriever")
        
        context_aware_doc_writer_agent = self._prelude | self.doc_writer_agent
        doc_writing_node = functools.partial(self._agent_node, agent=context_aware_doc_writer_agent, name="DocWriter")
        
        context_aware_note_taking_agent = self._prelude | self.note_taking_agent
        note_taking_node = functools.partial(self._agent_node, agent=context_aware_note_taking_agent, name="NoteTaker")

        # Build research graph
        research_graph = StateGraph(ResearchTeamState)
        research_graph.add_node("Search", search_node)
        research_graph.add_node("PaperInformationRetriever", research_node)
        research_graph.add_node("supervisor", self.research_supervisor)

        research_graph.add_edge("Search", "supervisor")
        research_graph.add_edge("PaperInformationRetriever", "supervisor")
        research_graph.add_conditional_edges(
            "supervisor",
            lambda x: x["next"],
            {"Search": "Search", "PaperInformationRetriever": "PaperInformationRetriever", "FINISH": END},
        )
        research_graph.set_entry_point("supervisor")
        research_chain = research_graph.compile()

        # Build authoring graph
        authoring_graph = StateGraph(DocWritingState)
        authoring_graph.add_node("DocWriter", doc_writing_node)
        authoring_graph.add_node("NoteTaker", note_taking_node)
        authoring_graph.add_node("supervisor", self.doc_supervisor)

        authoring_graph.add_edge("DocWriter", "supervisor")
        authoring_graph.add_edge("NoteTaker", "supervisor")
        authoring_graph.add_conditional_edges(
            "supervisor",
            lambda x: x["next"],
            {"DocWriter": "DocWriter", "NoteTaker": "NoteTaker", "FINISH": END},
        )
        authoring_graph.set_entry_point("supervisor")
        authoring_chain = authoring_graph.compile()

        # Create entry functions
        def enter_research_chain(message: str):
            return {"messages": [HumanMessage(content=message)]}

        def enter_authoring_chain(message: str):
            return {
                "messages": [HumanMessage(content=message)],
                "team_members": "DocWriter, NoteTaker",
            }

        # Wrap chains
        self.research_chain = enter_research_chain | research_chain
        self.authoring_chain = enter_authoring_chain | authoring_chain

        # Build super graph
        super_graph = StateGraph(State)
        
        def get_last_message(state: State) -> str:
            return state["messages"][-1].content

        def join_graph(response: dict):
            return {"messages": [response["messages"][-1]]}

        super_graph.add_node("Research team", get_last_message | self.research_chain | join_graph)
        super_graph.add_node("Blog writing team", get_last_message | self.authoring_chain | join_graph)
        
        supervisor_node = self._create_team_supervisor(
            self.llm,
            "You are Jay Guwalani's main supervisor tasked with managing a conversation between the"
            " following teams: {team_members}. Given the following user request,"
            " respond with the worker to act next. Each worker will perform a"
            " task and respond with their results and status. When finished,"
            " respond with FINISH.",
            ["Research team", "Blog writing team"],
        )
        
        super_graph.add_node("supervisor", supervisor_node)
        super_graph.add_edge("Research team", "supervisor")
        super_graph.add_edge("Blog writing team", "supervisor")
        super_graph.add_conditional_edges(
            "supervisor",
            lambda x: x["next"],
            {
                "Blog writing team": "Blog writing team",
                "Research team": "Research team",
                "FINISH": END,
            },
        )
        super_graph.set_entry_point("supervisor")
        self.super_graph = super_graph.compile()

    def process_request(self, message: str, recursion_limit: int = 150):
        """Process a user request through the multi-agent system."""
        results = []
        try:
            for s in self.super_graph.stream(
                {"messages": [HumanMessage(content=message)]},
                {"recursion_limit": recursion_limit},
            ):
                if "__end__" not in s:
                    results.append(s)
        except Exception as e:
            results.append({"error": f"Workflow error: {str(e)}"})
        return results

    def research_only(self, message: str, recursion_limit: int = 100):
        """Use only the research team."""
        results = []
        try:
            for s in self.research_chain.stream(message, {"recursion_limit": recursion_limit}):
                if "__end__" not in s:
                    results.append(s)
        except Exception as e:
            results.append({"error": f"Research error: {str(e)}"})
        return results

    def document_only(self, message: str, recursion_limit: int = 100):
        """Use only the document writing team."""
        results = []
        try:
            for s in self.authoring_chain.stream(message, {"recursion_limit": recursion_limit}):
                if "__end__" not in s:
                    results.append(s)
        except Exception as e:
            results.append({"error": f"Document error: {str(e)}"})
        return results

    def get_created_files(self):
        """Get list of files created in the working directory."""
        try:
            if self.working_directory.exists():
                return [f for f in self.working_directory.rglob("*") if f.is_file()]
            return []
        except Exception:
            return []


# Example usage and main function
def main():
    """Main function for command line usage."""
    import sys
    import getpass
    
    print("Multi-Agent RAG System by Jay Guwalani")
    print("=" * 50)
    
    # Get API keys
    try:
        openai_key = os.getenv("OPENAI_API_KEY") or getpass.getpass("OpenAI API Key: ")
        tavily_key = os.getenv("TAVILY_API_KEY") or getpass.getpass("Tavily API Key: ")
        
        # Initialize system
        print("\nInitializing system...")
        system = MultiAgentRAGSystem(openai_key, tavily_key)
        print("System initialized successfully!")
        
        if len(sys.argv) > 1:
            # Command line mode
            message = " ".join(sys.argv[1:])
            print(f"\nProcessing: {message}")
            results = system.process_request(message)
            
            for step in results:
                print(f"Step: {step}")
                
        else:
            # Interactive mode
            print("\nEnter your requests (type 'quit' to exit):")
            while True:
                try:
                    user_input = input("\n> ").strip()
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    if user_input:
                        results = system.process_request(user_input)
                        for step in results:
                            if "messages" in step.values().__iter__().__next__():
                                agent_name = list(step.keys())[0]
                                if agent_name != "supervisor":
                                    message = step[agent_name]["messages"][0]
                                    print(f"\n[{agent_name}]: {message.content}")
                                    
                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    break
                    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
