from typing import Dict, Any, List, Optional
import logging

# Google ADK imports
from google.adk.agents import SequentialAgent, ParallelAgent, ConditionalAgent, BaseAgent, Agent
from google.adk.events import Event, EventActions
from google.adk.agents.invocation_context import InvocationContext
from google.genai import types

# Set up logging
logger = logging.getLogger(__name__)

class DocumentValidationAgent(BaseAgent):
    """Validates the incoming document."""
    
    async def _run_async_impl(self, ctx: InvocationContext):
        logger.info("Validating document...")
        file_path = ctx.state.get("uploaded_file_path")
        document_content = ctx.state.get("document_content")
        
        if not file_path and not document_content:
            error = "No document path or content provided"
            return Event(
                author=self.name,
                actions=EventActions(state_delta={"validation_result": "invalid", "error": error}),
                content=types.Content(parts=[types.Part(text=error)])
            )
            
        return Event(
            author=self.name,
            actions=EventActions(state_delta={"validation_result": "valid"}),
            content=types.Content(parts=[types.Part(text="Document validated successfully")])
        )

class MetadataExtractionAgent(BaseAgent):
    """Extracts metadata from the document."""
    
    async def _run_async_impl(self, ctx: InvocationContext):
        logger.info("Extracting metadata...")
        try:
            # In a real implementation, extract metadata from the document
            metadata = {
                "title": "Sample Document",
                "authors": ["Author 1", "Author 2"],
                "year": 2025,
                "abstract": "This is a sample abstract."
            }
            
            return Event(
                author=self.name,
                actions=EventActions(state_delta={"metadata": metadata}),
                content=types.Content(parts=[types.Part(text="Metadata extracted successfully")])
            )
        except Exception as e:
            logger.exception("Error extracting metadata")
            return Event(
                author=self.name,
                actions=EventActions(state_delta={"error": f"Metadata extraction failed: {str(e)}"}),
                content=types.Content(parts=[types.Part(text=f"Error extracting metadata: {str(e)}")])
            )

class TextExtractionAgent(BaseAgent):
    """Extracts text content from the document."""
    
    async def _run_async_impl(self, ctx: InvocationContext):
        logger.info("Extracting text...")
        try:
            # In a real implementation, extract text from the document
            extracted_text = "This is the extracted text from the document."
            
            # Save as artifact
            text_part = types.Part(text=extracted_text)
            ctx.save_artifact("extracted_text.txt", text_part)
            
            return Event(
                author=self.name,
                actions=EventActions(
                    state_delta={"has_extracted_text": True},
                    artifact_delta={"extracted_text.txt": text_part}
                ),
                content=types.Content(parts=[types.Part(text="Text extracted successfully")])
            )
        except Exception as e:
            logger.exception("Error extracting text")
            return Event(
                author=self.name,
                actions=EventActions(state_delta={"error": f"Text extraction failed: {str(e)}"}),
                content=types.Content(parts=[types.Part(text=f"Error extracting text: {str(e)}")])
            )

class IngestionAgent(Agent):
    """Main ingestion agent that orchestrates the document processing pipeline."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="ingestion_agent")
        self.config = config or {}
        
        # Create the document processing pipeline using SequentialAgent
        self.pipeline = SequentialAgent(
            name="document_processing_pipeline",
            agents=[
                # Document validation
                DocumentValidationAgent(name="document_validator"),
                
                # Parallel processing of metadata and text extraction
                ParallelAgent(
                    name="parallel_extraction",
                    agents=[
                        MetadataExtractionAgent(name="metadata_extractor"),
                        TextExtractionAgent(name="text_extractor")
                    ],
                    max_concurrent=2
                ),
                
                # Conditional processing based on extraction results
                ConditionalAgent(
                    name="post_processing",
                    condition_fn=self._should_process_further,
                    if_true=SequentialAgent(
                        name="additional_processing",
                        agents=[
                            # Add additional processing steps here
                        ]
                    ),
                    if_false=None  # Skip additional processing
                )
            ],
            description="Document ingestion and processing pipeline"
        )
    
    def _should_process_further(self, ctx: InvocationContext) -> bool:
        """Determine if additional processing is needed."""
        # Add your custom logic here
        return ctx.state.get("has_extracted_text", False)
    
    async def _run_async_impl(self, ctx: InvocationContext):
        """Run the ingestion pipeline."""
        logger.info("Starting document ingestion pipeline...")
        
        try:
            # Process the document through the pipeline
            async for event in self.pipeline.run_async(ctx):
                # Yield events from the pipeline
                yield event
                
            # Final success event
            yield Event(
                author=self.name,
                actions=EventActions(state_delta={"ingestion_complete": True}),
                content=types.Content(parts=[types.Part(text="Document ingestion completed successfully")])
            )
            
        except Exception as e:
            logger.exception("Error in ingestion pipeline")
            yield Event(
                author=self.name,
                actions=EventActions(state_delta={"ingestion_complete": False, "error": str(e)}),
                content=types.Content(parts=[types.Part(text=f"Document ingestion failed: {str(e)}")])
            )

# Create the main ingestion agent instance
ingestion_agent = IngestionAgent()

class MetadataExtractionAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext):
        logger.info("Extracting metadata...")
        
        try:
            # Load the document artifact
            artifact_part = ctx.load_artifact("uploaded_document.pdf")
            
            # In a real implementation, you would extract metadata from the PDF
            # For now, we'll use placeholder metadata
            metadata = {
                "title": "Sample Document",
                "authors": ["Author 1", "Author 2"],
                "publication_date": "2025-01-01",
                "journal": "Sample Journal"
            }
            
            # Save metadata as state
            yield Event(
                author=self.name,
                actions=EventActions(state_delta={"document_metadata": metadata}),
                content=types.Content(parts=[types.Part(text=f"Metadata extracted: {metadata}")])
            )
        except Exception as e:
            error_msg = f"Metadata extraction failed: {str(e)}"
            yield Event(
                author=self.name,
                actions=EventActions(state_delta={"error": error_msg}),
                content=types.Content(parts=[types.Part(text=error_msg)])
            )

class TextExtractionAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext):
        logger.info("Extracting text...")
        
        try:
            # Load the document artifact
            artifact_part = ctx.load_artifact("uploaded_document.pdf")
            pdf_bytes = artifact_part.inline_data.data
            
            # In a real implementation, you would extract text from the PDF
            # For now, we'll use placeholder text
            extracted_text = "This is sample extracted text from the document."
            
            # Save extracted text as artifact
            text_part = types.Part(text=extracted_text)
            ctx.save_artifact("extracted_text.txt", text_part)
            
            yield Event(
                author=self.name,
                actions=EventActions(
                    state_delta={"extracted_text": "extracted_text.txt"},
                    artifact_delta={"extracted_text.txt": text_part}
                ),
                content=types.Content(parts=[types.Part(text="Text extracted successfully")])
            )
        except Exception as e:
            error_msg = f"Text extraction failed: {str(e)}"
            yield Event(
                author=self.name,
                actions=EventActions(state_delta={"error": error_msg}),
                content=types.Content(parts=[types.Part(text=error_msg)])
            )

class CitationExtractionAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext):
        logger.info("Extracting citations...")
        
        try:
            # Get extracted text
            extracted_text_artifact = ctx.session.state.get("extracted_text")
            text_part = ctx.load_artifact(extracted_text_artifact)
            text = text_part.text
            
            # In a real implementation, you would extract citations from the text
            # For now, we'll use placeholder citations
            citations = [
                {"text": "Citation 1", "authors": ["Author A"], "year": "2023"},
                {"text": "Citation 2", "authors": ["Author B"], "year": "2024"}
            ]
            
            yield Event(
                author=self.name,
                actions=EventActions(state_delta={"extracted_citations": citations}),
                content=types.Content(parts=[types.Part(text=f"Citations extracted: {len(citations)}")])
            )
        except Exception as e:
            error_msg = f"Citation extraction failed: {str(e)}"
            yield Event(
                author=self.name,
                actions=EventActions(state_delta={"error": error_msg}),
                content=types.Content(parts=[types.Part(text=error_msg)])
            )

class FigureTableExtractionAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext):
        logger.info("Extracting figures and tables...")
        
        try:
            # Load the document artifact
            artifact_part = ctx.load_artifact("uploaded_document.pdf")
            pdf_bytes = artifact_part.inline_data.data
            
            # In a real implementation, you would extract figures and tables from the PDF
            # For now, we'll use placeholder data
            figures_tables = {
                "figures": [{"id": "fig1", "caption": "Figure 1"}],
                "tables": [{"id": "tab1", "caption": "Table 1"}]
            }
            
            yield Event(
                author=self.name,
                actions=EventActions(state_delta={"extracted_figures_tables": figures_tables}),
                content=types.Content(parts=[types.Part(text="Figures and tables extracted")])
            )
        except Exception as e:
            error_msg = f"Figure/table extraction failed: {str(e)}"
            yield Event(
                author=self.name,
                actions=EventActions(state_delta={"error": error_msg}),
                content=types.Content(parts=[types.Part(text=error_msg)])
            )

class QualityValidationAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext):
        logger.info("Validating extraction quality...")
        
        try:
            # Check if all required data is present
            metadata = ctx.session.state.get("document_metadata")
            extracted_text = ctx.session.state.get("extracted_text")
            citations = ctx.session.state.get("extracted_citations")
            figures_tables = ctx.session.state.get("extracted_figures_tables")
            
            # Simple quality check
            quality = {
                "metadata_quality": 0.9 if metadata else 0.0,
                "text_quality": 0.8 if extracted_text else 0.0,
                "citations_quality": 0.7 if citations else 0.0,
                "figures_tables_quality": 0.6 if figures_tables else 0.0
            }
            
            # Calculate overall score
            overall_score = sum(quality.values()) / len(quality)
            quality["overall_score"] = overall_score
            
            yield Event(
                author=self.name,
                actions=EventActions(state_delta={"quality_assessment": quality}),
                content=types.Content(parts=[types.Part(text=f"Quality assessment completed: {overall_score:.2f}/1.0")])
            )
        except Exception as e:
            error_msg = f"Quality validation failed: {str(e)}"
            yield Event(
                author=self.name,
                actions=EventActions(state_delta={"error": error_msg}),
                content=types.Content(parts=[types.Part(text=error_msg)])
            )

# Create sub-agent instances
document_validation_agent = DocumentValidationAgent(name="document_validation_agent")
metadata_extraction_agent = MetadataExtractionAgent(name="metadata_extraction_agent")
text_extraction_agent = TextExtractionAgent(name="text_extraction_agent")
citation_extraction_agent = CitationExtractionAgent(name="citation_extraction_agent")
figure_table_extraction_agent = FigureTableExtractionAgent(name="figure_table_extraction_agent")
quality_validation_agent = QualityValidationAgent(name="quality_validation_agent")

# # Define the main sequential ingestion agent
# class SequentialIngestionAgent(BaseAgent):
#     """
#     A sequential agent that processes documents in a step-by-step workflow.
    
#     Workflow:
#     1. Document validation
#     2. Metadata extraction
#     3. Text extraction
#     4. Citation extraction 
#     5. Figure/table extraction
#     6. Quality validation
#     """
#     def __init__(self):
#         super().__init__(name="sequential_ingestion_agent")
#         self.sub_agents = [
#             document_validation_agent, 
#             metadata_extraction_agent,
#             text_extraction_agent,
#             citation_extraction_agent,
#             figure_table_extraction_agent,
#             quality_validation_agent
#         ]
    
#     async def run_sub_agent(self, ctx: InvocationContext, agent: BaseAgent, output_key: Optional[str] = None):
#         """Run a sub-agent and collect its events"""
#         async for event in agent._run_async_impl(ctx):
#             # Pass through the event to the parent context
#             yield event
            
#     async def _run_async_impl(self, ctx: InvocationContext):
#         try:
#             logger.info("Starting sequential document processing...")
            
#             # Step 1: Document validation
#             validation_events = []
#             async for event in self.run_sub_agent(ctx, document_validation_agent):
#                 validation_events.append(event)
#                 yield event
            
#             # Check validation result
#             validation_result = ctx.session.state.get("validation_result", "invalid")
#             if validation_result != "valid":
#                 error_msg = ctx.session.state.get("error", "Document validation failed")
#                 yield Event(
#                     author=self.name,
#                     actions=EventActions(state_delta={"ingestion_complete": False}),
#                     content=types.Content(parts=[types.Part(text=f"Ingestion stopped: {error_msg}")])
#                 )
#                 return
            
#             # Step 2: Metadata extraction
#             async for event in self.run_sub_agent(ctx, metadata_extraction_agent):
#                 yield event
            
#             # Step 3: Text extraction
#             async for event in self.run_sub_agent(ctx, text_extraction_agent):
#                 yield event
            
#             # Step 4: Citation extraction
#             async for event in self.run_sub_agent(ctx, citation_extraction_agent):
#                 yield event
            
#             # Step 5: Figure/table extraction
#             async for event in self.run_sub_agent(ctx, figure_table_extraction_agent):
#                 yield event
            
#             # Step 6: Quality validation
#             async for event in self.run_sub_agent(ctx, quality_validation_agent):
#                 yield event
            
#             # Finalize ingestion
#             yield Event(
#                 author=self.name,
#                 actions=EventActions(state_delta={"ingestion_complete": True}),
#                 content=types.Content(parts=[types.Part(text="Document ingestion completed successfully")])
#             )
            
#         except Exception as e:
#             error_message = f"Error processing document: {str(e)}"
#             yield Event(
#                 author=self.name,
#                 actions=EventActions(
#                     state_delta={"ingestion_complete": False, "error": error_message}
#                 ),
#                 content=types.Content(parts=[types.Part(text=error_message)])
#             )

# Create the sequential ingestion agent
sequential_ingestion_agent = SequentialIngestionAgent(name="sequential_ingestion_agent")