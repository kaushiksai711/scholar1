from google.cloud.aiplatform import Part, Blob
from google.cloud.aiplatform.gapic.schema import predict
from typing import Dict, Any, List, Optional
import json
import os
import logging
from datetime import datetime
import PyPDF2

logger = logging.getLogger(__name__)

async def extract_text_from_pdf(tool_context, file_path: str) -> Dict[str, Any]:
    """
    Extract text content from a PDF file and store it as an artifact.

    Args:
        tool_context: The tool context containing state and artifacts
        file_path: Path to the PDF file to extract text from

    Returns:
        Dict containing status, message, and artifact name if successful
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return {"status": "error", "message": f"File not found: {file_path}"}

        # Extract text from PDF
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""

        # Create artifact name based on input filename
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        artifact_name = f"{base_name}_text_{int(datetime.now().timestamp())}"
        
        # Create and save artifact
        text_artifact = Part(
            inline_data=Blob(
                mime_type="text/plain",
                data=text.encode('utf-8')
            )
        )
        tool_context.save_artifact(filename=artifact_name, artifact=text_artifact)
        
        return {
            "status": "success",
            "message": f"Successfully extracted text from {file_path}",
            "artifact_name": artifact_name
        }
        
    except Exception as e:
        error_msg = f"Error extracting text from {file_path}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"status": "error", "message": error_msg}

async def extract_metadata_from_pdf(tool_context, file_path: str) -> Dict[str, Any]:
    """
    Extract metadata from a PDF file and store it as an artifact.

    Args:
        tool_context: The tool context containing state and artifacts
        file_path: Path to the PDF file to extract metadata from

    Returns:
        Dict containing status, metadata, and artifact name if successful
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return {"status": "error", "message": f"File not found: {file_path}"}

        # Extract metadata from PDF
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            metadata = reader.metadata or {}
            
            # Convert PDF metadata to a serializable format
            metadata_dict = {}
            for key, value in metadata.items():
                if hasattr(value, 'encode'):
                    metadata_dict[key] = value
                else:
                    metadata_dict[key] = str(value)
            
            # Add basic file info
            metadata_dict["num_pages"] = len(reader.pages)
            metadata_dict["file_size"] = os.path.getsize(file_path)

        # Create artifact name based on input filename
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        artifact_name = f"{base_name}_metadata_{int(datetime.now().timestamp())}"
        
        # Create and save artifact
        metadata_artifact = Part(
            inline_data=Blob(
                mime_type="application/json",
                data=json.dumps(metadata_dict).encode('utf-8')
            )
        )
        tool_context.save_artifact(filename=artifact_name, artifact=metadata_artifact)
        
        return {
            "status": "success",
            "metadata": metadata_dict,
            "artifact_name": artifact_name
        }
        
    except Exception as e:
        error_msg = f"Error extracting metadata from {file_path}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"status": "error", "message": error_msg}

async def process_pdf_directory(tool_context, directory_path: str) -> Dict[str, Any]:
    """
    Process all PDF files in a directory, extract text and metadata, and store as artifacts.
    Creates an index artifact with information about all processed documents.

    Args:
        tool_context: The tool context containing state and artifacts
        directory_path: Path to the directory containing PDF files

    Returns:
        Dict containing status, message, and list of processed files
    """
    try:
        # Check if directory exists
        if not os.path.isdir(directory_path):
            return {"status": "error", "message": f"Directory not found: {directory_path}"}

        # Find all PDF files in the directory
        pdf_files = [f for f in os.listdir(directory_path) 
                    if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            return {"status": "error", "message": f"No PDF files found in {directory_path}"}

        processed_files = []
        text_artifacts = []
        
        # Process each PDF file
        for pdf_file in pdf_files:
            file_path = os.path.join(directory_path, pdf_file)
            
            # Extract text
            text_result = await extract_text_from_pdf(tool_context, file_path)
            
            # Extract metadata
            metadata_result = await extract_metadata_from_pdf(tool_context, file_path)
            
            if text_result["status"] == "success" and metadata_result["status"] == "success":
                processed_files.append({
                    "filename": pdf_file,
                    "text_artifact": text_result["artifact_name"],
                    "metadata_artifact": metadata_result["artifact_name"],
                    "metadata": metadata_result.get("metadata", {})
                })
                text_artifacts.append(text_result["artifact_name"])
        
        # Create directory index artifact
        index_data = {
            "directory_path": directory_path,
            "processed_at": datetime.now().isoformat(),
            "documents": [
                {
                    "filename": f["filename"],
                    "text_artifact": f["text_artifact"],
                    "metadata_artifact": f["metadata_artifact"],
                    "metadata": f["metadata"]
                }
                for f in processed_files
            ]
        }
        
        # Save directory index as artifact
        index_artifact_name = f"pdf_directory_index_{int(datetime.now().timestamp())}"
        index_artifact = Part(
            inline_data=Blob(
                mime_type="application/json",
                data=json.dumps(index_data).encode('utf-8')
            )
        )
        tool_context.save_artifact(filename=index_artifact_name, artifact=index_artifact)
        
        return {
            "status": "success",
            "message": f"Processed {len(processed_files)} PDF files from {directory_path}",
            "processed_files": processed_files,
            "index_artifact": index_artifact_name,
            "text_artifacts": text_artifacts
        }
        
    except Exception as e:
        error_msg = f"Error processing directory {directory_path}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"status": "error", "message": error_msg}