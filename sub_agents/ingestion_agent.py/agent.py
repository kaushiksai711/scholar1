from typing import Dict, Any, List, Optional
from google.cloud.aiplatform import Agent
from google.cloud.aiplatform.tool import FunctionTool
import logging

# Import local modules
from . import prompt
from . import tools

logger = logging.getLogger(__name__)

class IngestionAgent:
    """
    Ingestion Agent for processing PDF documents and extracting text and metadata.
    """
    
    def __init__(self, model: str = "gemini-2.0-flash"):
        """
        Initialize the Ingestion Agent.
        
        Args:
            model: The model to use for the agent
        """
        self.model = model
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """
        Create and configure the agent with its tools.
        
        Returns:
            Agent: Configured agent instance
        """
        # Create tool instances
        extract_text_tool = FunctionTool(
            tools.extract_text_from_pdf,
            name="extract_text_from_pdf",
            description="Extract text from a PDF file and store it as an artifact"
        )
        
        extract_metadata_tool = FunctionTool(
            tools.extract_metadata_from_pdf,
            name="extract_metadata_from_pdf",
            description="Extract metadata from a PDF file and store it as an artifact"
        )
        
        process_directory_tool = FunctionTool(
            tools.process_pdf_directory,
            name="process_pdf_directory",
            description="Process all PDF files in a directory, extract text and metadata"
        )
        
        # Create and return the agent
        return Agent(
            name="ingestion_agent",
            model=self.model,
            description="Agent responsible for ingesting and extracting structured information from research papers",
            instruction=prompt.INGESTION_AGENT_INSTRUCTION,
            tools=[
                extract_text_tool,
                extract_metadata_tool,
                process_directory_tool
            ]
        )
    
    async def process(self, tool_context, command: str) -> Dict[str, Any]:
        """
        Process a command using the ingestion agent.
        
        Args:
            tool_context: The tool context containing state and artifacts
            command: The command to process (e.g., path to a PDF or directory)
            
        Returns:
            Dict containing the processing results
        """
        try:
            # Determine if the command is a file or directory
            if os.path.isfile(command) and command.lower().endswith('.pdf'):
                # Process single PDF file
                text_result = await tools.extract_text_from_pdf(tool_context, command)
                metadata_result = await tools.extract_metadata_from_pdf(tool_context, command)
                
                if text_result["status"] == "success" and metadata_result["status"] == "success":
                    return {
                        "status": "success",
                        "message": f"Successfully processed PDF: {command}",
                        "text_artifact": text_result["artifact_name"],
                        "metadata_artifact": metadata_result["artifact_name"],
                        "metadata": metadata_result.get("metadata", {})
                    }
                else:
                    return {
                        "status": "error",
                        "message": f"Failed to process PDF {command}",
                        "text_result": text_result,
                        "metadata_result": metadata_result
                    }
                    
            elif os.path.isdir(command):
                # Process directory of PDFs
                return await tools.process_pdf_directory(tool_context, command)
                
            else:
                return {
                    "status": "error",
                    "message": f"Invalid command. Please provide a path to a PDF file or a directory containing PDFs."
                }
                
        except Exception as e:
            error_msg = f"Error processing command '{command}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"status": "error", "message": error_msg}

# Create a default instance of the agent
agent = IngestionAgent()