# API Reference - Multi-Agent RAG System

**Author:** Jay Guwalani  
**Version:** 1.0.0

## MultiAgentRAGSystem Class

### Constructor

```python
MultiAgentRAGSystem(openai_key: str, tavily_key: str)
```

Initialize the multi-agent system with required API keys.

**Parameters:**
- `openai_key` (str): OpenAI API key for LLM and embeddings
- `tavily_key` (str): Tavily API key for web search functionality

**Example:**
```python
system = MultiAgentRAGSystem(
    openai_key="sk-...",
    tavily_key="tvly-..."
)
```

### Methods

#### process_request

```python
process_request(message: str, recursion_limit: int = 150) -> List[Dict]
```

Process a user request through the complete multi-agent system.

**Parameters:**
- `message` (str): User request or query
- `recursion_limit` (int, optional): Maximum recursion depth. Default: 150

**Returns:**
- `List[Dict]`: List of workflow steps and agent responses

**Example:**
```python
results = system.process_request(
    "Research AI trends and write a technical report"
)
```

#### research_only

```python
research_only(message: str, recursion_limit: int = 100) -> List[Dict]
```

Use only the research team for information gathering.

**Parameters:**
- `message` (str): Research query
- `recursion_limit` (int, optional): Maximum recursion depth. Default: 100

**Returns:**
- `List[Dict]`: Research team workflow steps

**Example:**
```python
results = system.research_only(
    "What are the latest developments in RAG systems?"
)
```

#### document_only

```python
document_only(message: str, recursion_limit: int = 100) -> List[Dict]
```

Use only the document writing team for content creation.

**Parameters:**
- `message` (str): Document creation request
- `recursion_limit` (int, optional): Maximum recursion depth. Default: 100

**Returns:**
- `List[Dict]`: Document team workflow steps

**Example:**
```python
results = system.document_only(
    "Create a technical guide on multi-agent systems"
)
```

### Properties

#### working_directory

```python
@property
working_directory -> Path
```

Get the current working directory for file operations.

**Returns:**
- `Path`: Current working directory path

#### rag_chain

```python
@property
rag_chain -> Runnable
```

Access the RAG chain for direct document queries.

**Returns:**
- `Runnable`: LangChain RAG pipeline

## Agent Tools Reference

### Search Tools

#### TavilySearchResults

Web search functionality powered by Tavily API.

**Capabilities:**
- Real-time web search
- Current events and trends
- Market research data
- Technical documentation

**Usage:**
```python
# Automatically used by Search Agent
# Max results: 5 (configurable)
```

### RAG Tools

#### retrieve_information

```python
@tool
def retrieve_information(query: str) -> str
```

Retrieve information from the loaded document using RAG.

**Parameters:**
- `query` (str): Question about the document

**Returns:**
- `str`: Retrieved information with context

### Document Tools

#### create_outline

```python
@tool
def create_outline(points: List[str], file_name: str) -> str
```

Create and save a structured outline.

**Parameters:**
- `points` (List[str]): List of outline points
- `file_name` (str): Output file name

**Returns:**
- `str`: Confirmation message with file path

#### read_document

```python
@tool
def read_document(file_name: str, start: int = None, end: int = None) -> str
```

Read content from a specified document.

**Parameters:**
- `file_name` (str): Document file name
- `start` (int, optional): Start line number
- `end` (int, optional): End line number

**Returns:**
- `str`: Document content

#### write_document

```python
@tool
def write_document(content: str, file_name: str) -> str
```

Create and save a text document.

**Parameters:**
- `content` (str): Document content
- `file_name` (str): Output file name

**Returns:**
- `str`: Confirmation message with file path

#### edit_document

```python
@tool
def edit_document(file_name: str, inserts: Dict[int, str]) -> str
```

Edit an existing document by inserting text at specific lines.

**Parameters:**
- `file_name` (str): Document to edit
- `inserts` (Dict[int, str]): Line numbers (1-indexed) and text to insert

**Returns:**
- `str`: Confirmation message

## Agent Specifications

### Research Team

#### Search Agent

**Specialization:** Web-based information retrieval
**Tools:** `TavilySearchResults`
**Capabilities:**
- Current events research
- Market analysis
- Technical trends
- Comparative studies

#### RAG Agent (PaperInformationRetriever)

**Specialization:** Document-specific queries
**Tools:** `retrieve_information`
**Capabilities:**
- Academic paper analysis
- Technical specification extraction
- Detailed content queries
- Context-aware responses

#### Research Supervisor

**Role:** Research team coordination
**Routing Options:**
- `Search` - Web search tasks
- `PaperInformationRetriever` - Document queries
- `FINISH` - Complete research phase

### Document Team

#### Document Writer (DocWriter)

**Specialization:** Content creation and editing
**Tools:** `write_document`, `edit_document`, `read_document`
**Capabilities:**
- Technical writing
- Report generation
- Content editing
- Format standardization

#### Note Taker

**Specialization:** Structure and organization
**Tools:** `create_outline`, `read_document`
**Capabilities:**
- Outline creation
- Content summarization
- Structure planning
- Note organization

#### Document Supervisor

**Role:** Document team coordination
**Routing Options:**
- `DocWriter` - Content creation tasks
- `NoteTaker` - Structure and outline tasks
- `FINISH` - Complete document phase

### Meta-Supervisor

**Role:** System-wide coordination
**Routing Options:**
- `Research team` - Information gathering tasks
- `Blog writing team` - Content creation tasks
- `FINISH` - Complete workflow

## State Definitions

### Main System State

```python
class State(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str
```

### Research Team State

```python
class ResearchTeamState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: List[str]
    next: str
```

### Document Writing State

```python
class DocWritingState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: str
    next: str
    current_files: str
```

## Error Handling

### Common Exceptions

#### APIKeyError
Raised when API keys are missing or invalid.

```python
try:
    system = MultiAgentRAGSystem(openai_key, tavily_key)
except APIKeyError as e:
    print(f"API key error: {e}")
```

#### RecursionLimitError
Raised when recursion limit is exceeded.

```python
try:
    results = system.process_request(message, recursion_limit=50)
except RecursionLimitError as e:
    print(f"Recursion limit exceeded: {e}")
```

#### FileOperationError
Raised during file read/write operations.

```python
try:
    results = system.document_only("Create document")
except FileOperationError as e:
    print(f"File operation failed: {e}")
```

## Configuration Options

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key

# Optional
OPENAI_MODEL=gpt-4-1106-preview
EMBEDDING_MODEL=text-embedding-3-small
MAX_RECURSION_LIMIT=150
DEFAULT_CHUNK_SIZE=300
DEFAULT_CHUNK_OVERLAP=0
```

### Model Configuration

```python
# Custom LLM configuration
system.llm = ChatOpenAI(
    model="gpt-4-1106-preview",
    temperature=0.7,
    max_tokens=2000
)

# Custom embedding configuration
system.embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    chunk_size=1000
)
```

## Response Formats

### Standard Response Structure

```python
{
    "agent_name": {
        "messages": [
            {
                "content": "Agent response content",
                "name": "AgentName",
                "type": "human"
            }
        ]
    }
}
```

### Supervisor Response Structure

```python
{
    "supervisor": {
        "next": "AgentName" | "FINISH"
    }
}
```

## Performance Metrics

### Timing Benchmarks

- **RAG Query Response**: < 200ms
- **Web Search**: 1-3 seconds
- **Document Creation**: 2-5 seconds
- **Complete Workflow**: 30-60 seconds

### Resource Usage

- **Memory**: 1-2 GB typical usage
- **CPU**: Moderate during processing
- **Storage**: Temporary files only
- **Network**: API call dependent

## Best Practices

### Efficient Usage

1. **Batch Similar Requests**: Group related queries
2. **Set Appropriate Limits**: Use reasonable recursion limits
3. **Monitor Resources**: Track memory and API usage
4. **Handle Errors**: Implement proper error handling

### Performance Optimization

1. **Cache Results**: Reuse expensive computations
2. **Optimize Queries**: Use specific, focused requests
3. **Parallel Processing**: Leverage concurrent capabilities
4. **Resource Pooling**: Reuse connections and models

This API reference provides comprehensive documentation for integrating and extending the Multi-Agent RAG System developed by Jay Guwalani.
