"""
Ingestion Agent module for processing PDF documents and extracting text and metadata.
"""

from .agent import IngestionAgent, agent
from . import prompt, tools

__all__ = [
    'IngestionAgent',  # Agent class
    'agent',           # Default agent instance
    'prompt',          # Prompt module
    'tools'            # Tools module
]