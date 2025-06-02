from datetime import datetime
from typing import Dict, List, Any, Optional, Union, cast
import os
import json
import logging
from pathlib import Path

# Google ADK imports
from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext
from google.adk.artifacts import InMemoryArtifactService
from google.adk.artifacts.base_artifact_service import BaseArtifactService
from google.genai import types
from google.adk.models.lite_llm import LiteLlm
# Local imports
from .models import DocumentRequest, DocumentResponse

# Set up logging
logger = logging.getLogger(__name__)

# Import sub-agents
from .sub_agents.ingestion_agent.agent import ingestion_agent
# Import other sub-agents (to be implemented)
# from .sub_agents.citation_graph_agent.agent import citation_graph_agent
# from .sub_agents.cross_paper_analysis_agent.agent import cross_paper_analysis_agent
# from .sub_agents.deep_search_agent.agent import deep_search_agent
# from .sub_agents.insight_agent.agent import insight_agent
# from .sub_agents.visualization_agent.agent import visualization_agent

# Import tools
from .tools import (
    analyze_document_complexity,
    route_to_appropriate_agent,
    evaluate_agent_performance,
    process_user_feedback
)


class RouterAgent:
    """Main router agent for ScholarVerse that handles document processing and routing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the RouterAgent with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.artifact_service: BaseArtifactService = InMemoryArtifactService()
        self._initialize_state()
    
    def _initialize_state(self):
        """Initialize the agent's state."""
        self.state = {
            'documents': {},
            'sessions': {},
            'document_status': {},
            'analysis_results': {},
            'user_preferences': {},
            'interaction_history': []
        }
    
    def _get_tool_context(self, user_id: str) -> ToolContext:
        """Create a tool context for the given user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            ToolContext: Configured tool context
        """
        return ToolContext(
            user_id=user_id,
            state=self.state,
            artifact_service=self.artifact_service
        )
    
    async def process_document(self, request: DocumentRequest) -> DocumentResponse:
        """Process a document through the ingestion pipeline.
        
        Args:
            request: Document processing request
            
        Returns:
            DocumentResponse: Processing results
        """
        start_time = datetime.utcnow()
        session_id = request.options.get('session_id', f'sess_{int(start_time.timestamp())}')
        doc_id = f'doc_{int(start_time.timestamp())}'
        
        # Initialize context
        context = self._get_tool_context(request.user_id)
        
        try:
            # Process document using the ingestion agent
            result = await self._process_with_ingestion_agent(context, doc_id, session_id, request)
            
            # Update state
            self._update_document_state(doc_id, {
                'status': 'processed',
                'processed_at': datetime.utcnow().isoformat(),
                'processing_results': result
            })
            
            return DocumentResponse(
                status='success',
                document_id=doc_id,
                processing_results=result
            )
            
        except Exception as e:
            error_msg = f"Error processing document: {str(e)}"
            logger.exception(error_msg)
            
            # Update state with error
            self._update_document_state(doc_id, {
                'status': 'error',
                'error': error_msg,
                'failed_at': datetime.utcnow().isoformat()
            })
            
            return DocumentResponse(
                status='error',
                document_id=doc_id,
                error=error_msg
            )
    
    async def _process_with_ingestion_agent(
        self,
        context: ToolContext,
        doc_id: str,
        session_id: str,
        request: DocumentRequest
    ) -> Dict[str, Any]:
        """Process document using the ingestion agent."""
        # This is where we would call the ingestion agent's process_document method
        # For now, we'll implement a simplified version here
        
        # Save the original document
        content = await self._get_document_content(request)
        mime_type = request.options.get('mime_type', self._detect_mime_type(content, request.document_path))
        
        # Create and save artifact
        artifact = types.Part(
            inline_data=types.Blob(
                mime_type=mime_type,
                data=content if isinstance(content, bytes) else content.encode('utf-8')
            )
        )
        
        version = await self.artifact_service.save_artifact(
            app_name="scholarverse",
            user_id=context.user_id or "default_user",
            session_id=session_id,
            filename=f"{doc_id}/original",
            artifact=artifact
        )
        
        # Process based on content type
        processing_results = {}
        if mime_type == 'application/pdf':
            processing_results.update(await self._process_pdf(content, doc_id, context, session_id))
        
        return {
            'status': 'success',
            'document_id': doc_id,
            'session_id': session_id,
            'processing_steps': processing_results
        }
    
    async def _get_document_content(self, request: DocumentRequest) -> Union[bytes, str]:
        """Get document content from request."""
        if request.document_content is not None:
            return request.document_content
        elif request.document_path:
            with open(request.document_path, 'rb') as f:
                return f.read()
        else:
            raise ValueError("Either document_path or document_content must be provided")
    
    def _detect_mime_type(self, content: Union[bytes, str], file_path: Optional[str] = None) -> str:
        """Detect MIME type from content or file extension."""
        if file_path and file_path.lower().endswith('.pdf'):
            return 'application/pdf'
        return 'application/octet-stream'
    
    async def _process_pdf(
        self,
        content: bytes,
        doc_id: str,
        context: ToolContext,
        session_id: str
    ) -> Dict[str, Any]:
        """Process PDF document."""
        try:
            import fitz  # PyMuPDF
            import io
            
            # Extract text
            doc = fitz.open(stream=io.BytesIO(content), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            
            # Save extracted text
            text_artifact = types.Part(
                inline_data=types.Blob(
                    mime_type="text/plain",
                    data=text.encode('utf-8')
                )
            )
            
            await self.artifact_service.save_artifact(
                app_name="scholarverse",
                user_id=context.user_id or "default_user",
                session_id=session_id,
                filename=f"{doc_id}/extracted_text",
                artifact=text_artifact
            )
            
            # Extract basic metadata
            metadata = {
                'page_count': len(doc),
                'extracted_at': datetime.utcnow().isoformat()
            }
            
            metadata_artifact = types.Part(
                inline_data=types.Blob(
                    mime_type="application/json",
                    data=json.dumps(metadata).encode('utf-8')
                )
            )
            
            await self.artifact_service.save_artifact(
                app_name="scholarverse",
                user_id=context.user_id or "default_user",
                session_id=session_id,
                filename=f"{doc_id}/metadata",
                artifact=metadata_artifact
            )
            
            return {
                'text_extraction': {
                    'status': 'success',
                    'characters_extracted': len(text)
                },
                'metadata_extraction': {
                    'status': 'success',
                    'fields_extracted': list(metadata.keys())
                }
            }
            
        except Exception as e:
            logger.exception("Error processing PDF")
            return {
                'text_extraction': {
                    'status': 'error',
                    'error': str(e)
                }
            }
    
    def _update_document_state(self, doc_id: str, updates: Dict[str, Any]):
        """Update document state with the given updates."""
        if 'documents' not in self.state:
            self.state['documents'] = {}
        
        if doc_id not in self.state['documents']:
            self.state['documents'][doc_id] = {}
        
        self.state['documents'][doc_id].update(updates)
        
        # Ensure timestamps
        if 'updated_at' not in self.state['documents'][doc_id]:
            self.state['documents'][doc_id]['updated_at'] = datetime.utcnow().isoformat()
        if 'created_at' not in self.state['documents'][doc_id]:
            self.state['documents'][doc_id]['created_at'] = datetime.utcnow().isoformat()

# Create the main router agent
root_agent = Agent(
    name="root_agent",
    model="gemini-2.0-flash",  # Or your preferred model
    description="Main orchestrator agent for ScholarVerse that routes tasks to specialized sub-agents",
    instruction="""
    You are the main Router Agent for ScholarVerse, responsible for coordinating the flow of information 
    between specialized sub-agents that handle different aspects of scientific paper analysis.

    **State Management:**
    - Track document processing status in state['document_status']
    - Maintain analysis results in state['analysis_results']
    - Store user preferences in state['user_preferences']
    - Track interaction history in state['interaction_history']

    **Available Sub-Agents:**
    1. Ingestion Agent: Processes and extracts information from uploaded documents
    2. Citation Graph Agent: Builds and analyzes citation networks
    3. Cross-Paper Analysis Agent: Performs comparative analysis across multiple papers
    4. Deep Search Agent: Conducts web research for additional context
    5. Insight Agent: Generates AI-powered insights and summaries
    6. Visualization Agent: Creates interactive visualizations of the knowledge graph

    **Workflow:**
    1. When a new document is uploaded, route it to the Ingestion Agent
    2. After ingestion, coordinate analysis with appropriate sub-agents
    3. Use the analysis results to build a comprehensive knowledge graph
    4. Generate insights and visualizations based on user queries

    **State Structure:**
    {
        'document_status': {
            'ingested': bool,
            'processed': bool,
            'analyzed': bool
        },
        'analysis_results': {
            'key_concepts': List[str],
            'relationships': List[Dict],
            'citations': List[Dict]
        },
        'user_preferences': {
            'preferred_topics': List[str],
            'analysis_depth': str  # 'basic', 'standard', 'advanced'
        },
        'interaction_history': List[Dict]
    }

    Always maintain context and state between interactions to provide coherent responses.
    
    **Document Processing Workflow:**
    1. When a document is uploaded, it's saved to the uploads directory
    2. The router agent analyzes the document and routes it to the ingestion agent
    3. The ingestion agent processes the document and extracts relevant information
    4. The results are stored in the agent's state and can be accessed by other agents
    """,
    sub_agents=[
        ingestion_agent
    ],
    tools=[
        analyze_document_complexity,
        route_to_appropriate_agent,
        evaluate_agent_performance,
        process_user_feedback
    ]
)
