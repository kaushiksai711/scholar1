from typing import Dict, Any, Optional, Union
from pydantic import BaseModel

class DocumentRequest(BaseModel):
    """Request model for document processing."""
    document_path: Optional[str] = None
    document_content: Optional[Union[str, bytes]] = None
    user_id: str = "default_user"
    options: Dict[str, Any] = {}

class DocumentResponse(BaseModel):
    """Response model for document processing."""
    status: str  # 'success', 'error', or 'partial_success'
    document_id: Optional[str] = None
    error: Optional[str] = None
    processing_results: Optional[Dict[str, Any]] = None
