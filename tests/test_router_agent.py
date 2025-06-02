import os
import sys
import asyncio
from pathlib import Path
import pytest
from google.genai import types

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scholar_verse.agents.router_agent.agent import RouterAgent
from scholar_verse.agents.router_agent.models import DocumentRequest

# Test PDF file path (create a simple PDF for testing if it doesn't exist)
TEST_PDF_PATH = "test_document.pdf"

def create_test_pdf():
    """Create a simple test PDF file if it doesn't exist."""
    if not os.path.exists(TEST_PDF_PATH):
        import fitz  # PyMuPDF
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "This is a test PDF document.")
        doc.save(TEST_PDF_PATH)
        doc.close()

@pytest.fixture
def router_agent():
    """Fixture to create a RouterAgent instance for testing."""
    return RouterAgent()

@pytest.fixture(autouse=True)
def setup():
    """Setup test environment."""
    create_test_pdf()
    yield
    # Cleanup if needed

@pytest.mark.asyncio
async def test_process_document_from_path(router_agent):
    """Test processing a document from a file path."""
    request = DocumentRequest(
        document_path=TEST_PDF_PATH,
        user_id="test_user",
        options={"mime_type": "application/pdf"}
    )
    
    response = await router_agent.process_document(request)
    
    assert response.status == "success"
    assert response.document_id is not None
    assert "text_extraction" in response.processing_results["processing_steps"]
    assert response.processing_results["processing_steps"]["text_extraction"]["status"] == "success"
    assert response.processing_results["processing_steps"]["text_extraction"]["characters_extracted"] > 0

@pytest.mark.asyncio
async def test_process_document_from_content(router_agent):
    """Test processing a document from in-memory content."""
    test_content = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj\n4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n5 0 obj<</Length 44>>stream\nBT\n/F1 24 Tf\n100 700 Td\n(Hello, World!) Tj\nET\nendstream\nendobj\nxref\n0 6\n0000000000 65535 f \n0000000015 00000 n \n0000000069 00000 n \n0000000128 00000 n \n0000000218 00000 n \n0000000247 00000 n \ntrailer<</Size 6/Root 1 0 R>>\nstartxref\n377\n%%EOF"
    
    request = DocumentRequest(
        document_content=test_content,
        user_id="test_user",
        options={"mime_type": "application/pdf", "filename": "test.pdf"}
    )
    
    response = await router_agent.process_document(request)
    
    assert response.status == "success"
    assert response.document_id is not None
    assert "text_extraction" in response.processing_results["processing_steps"]

@pytest.mark.asyncio
async def test_artifact_storage(router_agent):
    """Test that artifacts are properly stored and can be retrieved."""
    # First process a document
    request = DocumentRequest(
        document_path=TEST_PDF_PATH,
        user_id="test_user",
        options={"mime_type": "application/pdf"}
    )
    response = await router_agent.process_document(request)
    
    # Now try to retrieve the stored artifacts
    doc_id = response.document_id
    session_id = response.processing_results["session_id"]
    
    # Get the original artifact
    original_artifact = await router_agent.artifact_service.load_artifact(
        app_name="scholarverse",
        user_id="test_user",
        session_id=session_id,
        filename=f"{doc_id}/original"
    )
    
    assert original_artifact is not None
    assert hasattr(original_artifact, 'inline_data')
    assert original_artifact.inline_data.mime_type == "application/pdf"
    
    # Get the extracted text
    text_artifact = await router_agent.artifact_service.load_artifact(
        app_name="scholarverse",
        user_id="test_user",
        session_id=session_id,
        filename=f"{doc_id}/extracted_text"
    )
    
    assert text_artifact is not None
    assert hasattr(text_artifact, 'inline_data')
    assert text_artifact.inline_data.mime_type == "text/plain"
    assert len(text_artifact.inline_data.data) > 0

if __name__ == "__main__":
    # Create a test PDF if it doesn't exist
    create_test_pdf()
    
    # Run the tests
    import pytest
    pytest.main(["-v", "test_router_agent.py"])
