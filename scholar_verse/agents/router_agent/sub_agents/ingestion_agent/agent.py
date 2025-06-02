from google.adk.agents import Agent
from typing import Dict, Any, List, Optional

# Import tools
from .tools import (
    extract_metadata,
    extract_text_from_pdf,
    extract_citations,
    extract_figures_tables,
    validate_extraction_quality,
    process_document
)

# Import prompts
from .prompt import INGESTION_AGENT_INSTRUCTION

# Create the ingestion agent
ingestion_agent = Agent(
    name="ingestion_agent",
    model="gemini-2.0-flash",  # Or your preferred model
    description="Agent responsible for ingesting and extracting structured information from scientific documents",
    instruction=INGESTION_AGENT_INSTRUCTION,
    tools=[
        extract_metadata,
        extract_text_from_pdf,
        extract_citations,
        extract_figures_tables,
        validate_extraction_quality,
        process_document
    ]
)