# ScholarVerse: Autonomous Scientific Knowledge Graph Platform

## Overview

ScholarVerse is an advanced knowledge graph platform for scientific literature that leverages Google's Agent Development Kit (ADK) to orchestrate autonomous intelligent agents for processing, analyzing, and visualizing scientific papers. The platform ingests scientific papers, extracts key information, builds a knowledge graph of citations and relationships, generates insights, performs real-time deep search for related concepts, conducts cross-paper analysis, and visualizes the connections in an interactive 3D environment.

## Key Features

- **Autonomous Multi-Agent Architecture**: Utilizes Google's Agent Development Kit (ADK) to coordinate self-directed, adaptive agents that make decisions based on context and feedback
- **Intelligent Document Processing**: Extracts text, metadata, and citations from scientific papers
- **Advanced Knowledge Graph Construction**: Builds a comprehensive graph of papers, authors, citations with deep interlinking of methodologies, findings, and concepts
- **Cross-Paper Analysis**: Performs comparative analysis across multiple papers to identify similarities, differences, and trends in methodologies and findings
- **Real-Time Deep Search**: Provides LLM-powered web scraping to find additional information about concepts, methodologies, and authors mentioned in papers
- **AI-Powered Insights**: Generates summaries, trend analyses, and comparative analyses using Google Gemini
- **Interactive 3D Visualization**: Visualizes the knowledge graph in an immersive 3D environment with concept clustering and relationship highlighting

## System Architecture

ScholarVerse is built on a multi-agent architecture using Google's Agent Development Kit (ADK), with each agent having its own specialized functionality:

### Agent Structure

Each agent follows a standardized structure:
- `agent.py` - Core agent implementation using ADK
- `prompt.py` - Agent-specific prompts and instructions
- `tools.py` - Functions and utilities used by the agent
- `sub_agents/` - Folder containing specialized sub-agents (for main agents only)

### Agent Directory Structure

```
scholar_verse/
├── agents/
│   ├── router_agent/                    # Main Orchestrator Agent
│   │   ├── agent.py                     # Adaptive Router Agent implementation
│   │   ├── prompt.py                    # Router coordination prompts
│   │   ├── tools.py                     # Dynamic routing and evaluation functions
│   │   └── sub_agents/
│   │       ├── ingestion_agent/         # Document Processing Sub-Agent
│   │       │   ├── agent.py             # Ingestion agent implementation
│   │       │   ├── prompt.py            # Document processing prompts
│   │       │   └── tools.py             # PDF extraction and processing functions
│   │       ├── citation_graph_agent/    # Knowledge Graph Sub-Agent
│   │       │   ├── agent.py             # Citation graph agent implementation
│   │       │   ├── prompt.py            # Graph construction prompts
│   │       │   └── tools.py             # Neo4j operations and graph analysis functions
│   │       ├── cross_paper_analysis_agent/ # Comparative Analysis Sub-Agent
│   │       │   ├── agent.py             # Cross-paper analysis implementation
│   │       │   ├── prompt.py            # Comparative analysis prompts
│   │       │   └── tools.py             # Methodology comparison and trend analysis functions
│   │       ├── deep_search_agent/       # Web Research Sub-Agent
│   │       │   ├── agent.py             # Deep search agent implementation
│   │       │   ├── prompt.py            # Web research prompts
│   │       │   └── tools.py             # Web scraping and content validation functions
│   │       ├── insight_agent/           # Intelligence Generation Sub-Agent
│   │       │   ├── agent.py             # Insight generation implementation
│   │       │   ├── prompt.py            # Insight generation prompts
│   │       │   └── tools.py             # Gemini integration and refinement functions
│   │       └── visualization_agent/     # 3D Visualization Sub-Agent
│   │           ├── agent.py             # Visualization agent implementation
│   │           ├── prompt.py            # Visualization prompts
│   │           └── tools.py             # Clustering and 3D graph generation functions
│   └── __init__.py
├── config/                              # Configuration files
├── database/                            # Database schemas and migrations
├── tests/                               # Unit and integration tests
├── deployment/                          # Deployment scripts
├── eval/                                # Evaluation scripts and metrics
└── web_research_cache/                  # Cache for web research results
```

## Agent Implementations

### 1. Router Agent (Main Orchestrator)

**Location**: `agents/router_agent/`

The Router Agent coordinates all sub-agents and manages the overall workflow.

**Files**:
- `agent.py` - Implements the main orchestration logic using ADK's Agent framework
- `prompt.py` - Contains prompts for dynamic routing decisions and workflow coordination
- `tools.py` - Functions for:
  - `analyze_document_complexity()` - Determines processing requirements
  - `route_to_appropriate_agent()` - Selects optimal agent for tasks
  - `evaluate_agent_performance()` - Monitors and optimizes agent performance
  - `process_user_feedback()` - Incorporates feedback into routing decisions

### 2. Ingestion Agent (Document Processing Sub-Agent)

**Location**: `agents/router_agent/sub_agents/ingestion_agent/`

Processes PDF documents and extracts structured information.

**Files**:
- `agent.py` - Implements document processing workflows
- `prompt.py` - Prompts for text extraction and metadata identification
- `tools.py` - Functions for:
  - `extract_pdf_text()` - Extracts text content from PDFs
  - `extract_metadata()` - Identifies authors, titles, publication info
  - `parse_document_structure()` - Identifies sections, figures, tables
  - `validate_extraction_quality()` - Assesses extraction completeness
  - `improve_extraction_methods()` - Self-learning extraction optimization

### 3. Citation Graph Agent (Knowledge Graph Sub-Agent)

**Location**: `agents/router_agent/sub_agents/citation_graph_agent/`

Builds and maintains the knowledge graph in Neo4j.

**Files**:
- `agent.py` - Implements graph construction and querying logic
- `prompt.py` - Prompts for relationship identification and graph analysis
- `tools.py` - Functions for:
  - `create_paper_nodes()` - Creates paper entities in Neo4j
  - `extract_citations()` - Identifies and extracts citation references
  - `build_citation_relationships()` - Establishes citation links
  - `analyze_research_clusters()` - Identifies related research groups
  - `execute_graph_queries()` - Processes natural language queries using NL2SQL

### 4. Cross-Paper Analysis Agent (Comparative Analysis Sub-Agent)

**Location**: `agents/router_agent/sub_agents/cross_paper_analysis_agent/`

Performs comparative analysis across multiple research papers.

**Files**:
- `agent.py` - Implements comparative analysis workflows
- `prompt.py` - Prompts for methodology comparison and trend identification
- `tools.py` - Functions for:
  - `compare_methodologies()` - Analyzes research approaches across papers
  - `identify_research_trends()` - Detects emerging patterns and topics
  - `analyze_conflicting_findings()` - Identifies contradictory results
  - `extract_common_concepts()` - Finds shared research concepts
  - `generate_trend_reports()` - Creates comparative analysis summaries

### 5. Deep Search Agent (Web Research Sub-Agent)

**Location**: `agents/router_agent/sub_agents/deep_search_agent/`

Conducts real-time web research to find additional information about concepts mentioned in papers.

**Files**:
- `agent.py` - Implements web research and information retrieval
- `prompt.py` - Prompts for search query generation and content validation
- `tools.py` - Functions for:
  - `search_academic_databases()` - Queries scholarly databases
  - `scrape_web_content()` - Extracts content from web sources
  - `validate_information_credibility()` - Assesses source reliability
  - `integrate_external_knowledge()` - Incorporates findings into knowledge graph
  - `semantic_search_vertex_ai()` - Performs semantic search using Vertex AI RAG

### 6. Insight Agent (Intelligence Generation Sub-Agent)

**Location**: `agents/router_agent/sub_agents/insight_agent/`

Generates AI-powered insights and summaries using Google Gemini.

**Files**:
- `agent.py` - Implements insight generation and refinement workflows
- `prompt.py` - Prompts for insight generation and quality improvement
- `tools.py` - Functions for:
  - `generate_paper_summaries()` - Creates concise paper summaries
  - `identify_research_gaps()` - Finds areas needing further research
  - `generate_trend_analysis()` - Analyzes research field developments
  - `refine_insights_iteratively()` - Implements critic/reviser pattern
  - `extract_key_claims()` - Identifies and verifies important statements
  - `integrate_gemini_analysis()` - Leverages Google Gemini for deep analysis

### 7. Visualization Agent (3D Visualization Sub-Agent)

**Location**: `agents/router_agent/sub_agents/visualization_agent/`

Prepares data for interactive 3D visualization of the knowledge graph.

**Files**:
- `agent.py` - Implements visualization data preparation
- `prompt.py` - Prompts for layout optimization and relationship highlighting
- `tools.py` - Functions for:
  - `cluster_related_concepts()` - Groups conceptually similar nodes
  - `optimize_3d_layout()` - Arranges nodes for optimal visualization
  - `highlight_important_relationships()` - Emphasizes key connections
  - `generate_three_js_data()` - Prepares data for Three.js rendering
  - `create_interactive_controls()` - Implements zoom, filter, and search features

## Technology Stack

- **Backend**: Python, FastAPI, Google ADK
- **Database**: Neo4j (graph database), Qdrant (vector search), BigQuery (for analytics)
- **AI/ML**: Google Gemini API, Document AI, Vertex AI RAG Engine
- **Agent Tools**: Web search, NL2SQL, Code Interpreter, Vertex AI Search
- **Evaluation Framework**: ADK AgentEvaluator for quality assessment
- **Frontend**: React, Three.js for 3D visualization
- **Cloud**: Google Cloud (GCS, Vertex AI, BigQuery)
- **Deployment**: Vertex AI Agent Engine

## Google ADK Integration

ScholarVerse leverages Google's Agent Development Kit (ADK) to create a powerful, scalable, and maintainable multi-agent system.

### Agent Implementation Pattern

Each agent follows this implementation pattern:

```python
# Example: agents/router_agent/sub_agents/ingestion_agent/agent.py
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from .tools import extract_pdf_text, extract_metadata, validate_extraction_quality
from .prompt import INGESTION_PROMPTS

class IngestionAgent(LlmAgent):
    def __init__(self):
        super().__init__(
            name="ingestion_agent",
            description="Processes PDF documents and extracts structured information",
            model="gemini-1.5-pro",
            instruction=INGESTION_PROMPTS["main_instruction"],
            tools=[
                FunctionTool(extract_pdf_text),
                FunctionTool(extract_metadata),
                FunctionTool(validate_extraction_quality)
            ]
        )
```

### Tool Implementation Pattern

Each agent's tools are implemented as functions:

```python
# Example: agents/router_agent/sub_agents/ingestion_agent/tools.py
import PyPDF2
from typing import Dict, List

def extract_pdf_text(pdf_path: str) -> Dict[str, str]:
    """Extract text content from PDF documents."""
    # Implementation here
    return {"text": extracted_text, "pages": page_count}

def extract_metadata(pdf_path: str) -> Dict[str, str]:
    """Extract metadata from PDF documents."""
    # Implementation here
    return {"title": title, "authors": authors, "publication_date": date}

def validate_extraction_quality(extracted_data: Dict) -> Dict[str, float]:
    """Assess the quality of extracted information."""
    # Implementation here
    return {"completeness_score": 0.95, "accuracy_score": 0.92}
```

### Prompt Management Pattern

Each agent's prompts are centralized:

```python
# Example: agents/router_agent/sub_agents/ingestion_agent/prompt.py
INGESTION_PROMPTS = {
    "main_instruction": """
    You are an expert document processing agent specialized in extracting 
    structured information from scientific papers. Your goal is to...
    """,
    "text_extraction": """
    Extract the main text content from this PDF, paying attention to...
    """,
    "metadata_extraction": """
    Identify and extract the following metadata from this document...
    """
}
```

## Autonomous Workflows

ScholarVerse supports the following autonomous workflows:

1. **Document Ingestion Workflow**:
   - Upload PDF → Extract text/metadata → Build citation graph → Generate summary

2. **Deep Research Workflow**:
   - Identify key concepts → Perform web search → Validate information → Integrate findings

3. **Cross-Paper Analysis Workflow**:
   - Select papers → Compare methodologies → Identify trends → Generate insights

4. **Visualization Workflow**:
   - Analyze graph structure → Cluster concepts → Generate 3D layout → Create interactions

## Setup and Installation

### Prerequisites

- Python 3.9+
- Google Cloud SDK
- Neo4j Aura DB (free tier available)
- Vertex AI enabled GCP project

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/scholar-verse.git
   cd scholar-verse
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install google-adk
   ```

4. **Configuration**:
   Create a `.env` file:
   ```
   GOOGLE_CLOUD_PROJECT=your-project-id
   NEO4J_URI=neo4j+s://your-neo4j-instance.databases.neo4j.io
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your-password
   VERTEX_AI_LOCATION=us-central1
   GEMINI_API_KEY=your-gemini-api-key
   ```

5. **Initialize the system**:
   ```bash
   python -m agents.router_agent.agent --init
   ```

### Running the Application

1. **Start backend services**:
   ```bash
   # Start Neo4j and other services
   docker-compose up -d
   
   # Run the main agent system
   python -m agents.router_agent.agent --serve
   ```

2. **Start the API server**:
   ```bash
   uvicorn api.main:app --reload
   ```

3. **Start the frontend**:
   ```bash
   cd frontend
   npm install
   npm start
   ```

## 10-Phase Implementation Plan

### Phase 1: Foundation and Router Agent
- Set up project structure
- Implement Router Agent (agent.py, prompt.py, tools.py)
- Create basic agent coordination

### Phase 2: Ingestion Agent
- Implement Ingestion Agent with PDF processing
- Create document extraction tools
- Set up quality validation

### Phase 3: Citation Graph Agent
- Implement Citation Graph Agent
- Set up Neo4j integration tools
- Create graph construction workflows

### Phase 4: Cross-Paper Analysis Agent
- Implement comparative analysis agent
- Create methodology comparison tools
- Set up trend identification

### Phase 5: Deep Search Agent
- Implement web research agent
- Create web scraping tools
- Set up information validation

### Phase 6: Insight Agent
- Implement insight generation agent
- Create Gemini integration tools
- Set up refinement workflows

### Phase 7: Visualization Agent
- Implement visualization agent
- Create 3D graph generation tools
- Set up clustering algorithms

### Phase 8: Frontend Integration
- Create React interface
- Implement 3D visualization component
- Set up user interaction workflows

### Phase 9: Testing and Evaluation
- Implement comprehensive testing
- Set up evaluation frameworks
- Create performance monitoring

### Phase 10: Deployment and Optimization
- Deploy to Google Cloud
- Optimize performance
- Create documentation

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow the agent structure pattern (agent.py, prompt.py, tools.py)
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google ADK Team for the Agent Development Kit
- Neo4j for the graph database
- The open-source community for countless libraries and tools

---

*ScholarVerse - Empowering Research Through AI*