from typing import Dict, Any, List, Optional, Tuple, Union
import re
import fitz  # PyMuPDF
from datetime import datetime
from google.adk.tools.tool_context import ToolContext

def extract_metadata(tool_context: ToolContext, document_content: str) -> Dict[str, Any]:
    """
    Extract metadata from the document content.
    
    Args:
        tool_context: The context containing state and other utilities
        document_content: The content of the document to process
        
    Returns:
        Dict containing extracted metadata
    """
    # Initialize state if needed
    if 'extracted_data' not in tool_context.state:
        tool_context.state['extracted_data'] = {}
    if 'document_status' not in tool_context.state:
        tool_context.state['document_status'] = {}
    if 'processing_history' not in tool_context.state:
        tool_context.state['processing_history'] = []
    
    # Simple metadata extraction (can be enhanced with more sophisticated methods)
    metadata = {
        'title': '',
        'authors': [],
        'abstract': '',
        'keywords': [],
        'publication_date': '',
        'doi': '',
        'extraction_timestamp': datetime.now().isoformat()
    }
    
    # Try to extract title (first non-empty line)
    lines = [line.strip() for line in document_content.split('\n') if line.strip()]
    if lines:
        metadata['title'] = lines[0]
    
    # Try to extract authors (simple pattern matching)
    if len(lines) > 1:
        # Look for author patterns (e.g., "Author1, A.; Author2, B.")
        author_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]*\.?)(?:\s+[A-Z][a-z]+)*)'
        potential_authors = re.findall(author_pattern, lines[1])
        if potential_authors:
            metadata['authors'] = [auth.strip() for auth in potential_authors[:10]]  # Limit to first 10 authors
    
    # Try to find abstract
    abstract_start = -1
    for i, line in enumerate(lines):
        if 'abstract' in line.lower() and len(line.split()) < 3:  # Simple check for abstract heading
            abstract_start = i + 1
            break
    
    if abstract_start > 0 and abstract_start < len(lines):
        abstract = []
        for line in lines[abstract_start:]:
            if len(line.split()) > 1:  # Non-empty line
                abstract.append(line.strip())
            else:
                break
        metadata['abstract'] = ' '.join(abstract)
    
    # Update state
    tool_context.state['extracted_data']['metadata'] = metadata
    tool_context.state['document_status']['metadata_extracted'] = True
    
    # Log this action
    tool_context.state['processing_history'].append({
        'action': 'metadata_extraction',
        'timestamp': datetime.now().isoformat(),
        'status': 'success',
        'metadata_fields_extracted': list(metadata.keys())
    })
    
    return {
        'status': 'success',
        'metadata': metadata
    }

def extract_text_from_pdf(tool_context: ToolContext, pdf_path: str) -> Dict[str, Any]:
    """
    Extract text from a PDF document.
    
    Args:
        tool_context: The context containing state and other utilities
        pdf_path: Path to the PDF file (can be relative or absolute)
        
    Returns:
        Dict containing extracted text and status
    """
    # Initialize state if needed
    if 'extracted_data' not in tool_context.state:
        tool_context.state['extracted_data'] = {}
    if 'document_status' not in tool_context.state:
        tool_context.state['document_status'] = {}
    if 'processing_history' not in tool_context.state:
        tool_context.state['processing_history'] = []
    
    # Convert to absolute path if it's not already
    if not os.path.isabs(pdf_path):
        pdf_path = os.path.abspath(pdf_path)
        
    # Verify the file exists and is a PDF
    if not os.path.exists(pdf_path):
        error_msg = f'PDF file not found: {pdf_path}'
        tool_context.state['processing_history'].append({
            'action': 'pdf_extraction',
            'timestamp': datetime.now().isoformat(),
            'status': 'error',
            'error': error_msg,
            'file_path': pdf_path,
            'current_working_directory': os.getcwd()
        })
        return {
            'status': 'error',
            'message': error_msg,
            'current_working_directory': os.getcwd()
        }
        
    if not pdf_path.lower().endswith('.pdf'):
        error_msg = 'File is not a PDF'
        tool_context.state['processing_history'].append({
            'action': 'pdf_extraction',
            'timestamp': datetime.now().isoformat(),
            'status': 'error',
            'error': error_msg,
            'file_path': pdf_path
        })
        return {
            'status': 'error',
            'message': error_msg,
            'file_path': pdf_path
        }
        
    # Log the extraction attempt
    tool_context.state['processing_history'].append({
        'action': 'pdf_extraction',
        'timestamp': datetime.now().isoformat(),
        'file_path': pdf_path,
        'status': 'started'
    })
    
    doc = None
    try:
        # Open the PDF file with error handling for corrupted files
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            error_msg = f'Failed to open PDF file: {str(e)}'
            tool_context.state['processing_history'].append({
                'action': 'text_extraction',
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': error_msg,
                'file_path': pdf_path
            })
            return {
                'status': 'error',
                'message': error_msg,
                'document_path': pdf_path
            }
            
        text = ""
        pages_processed = 0
        
        # Extract text from each page with error handling per page
        for page_num in range(len(doc)):
            try:
                page = doc.load_page(page_num)
                page_text = page.get_text()
                if page_text:
                    text += page_text + "\n\n"  # Add some space between pages
                    pages_processed += 1
            except Exception as e:
                # Log page error but continue with other pages
                tool_context.state['processing_history'].append({
                    'action': 'text_extraction_page',
                    'timestamp': datetime.now().isoformat(),
                    'status': 'partial_error',
                    'error': f'Error processing page {page_num + 1}: {str(e)}',
                    'file_path': pdf_path,
                    'page': page_num + 1
                })
        
        if not text.strip():
            error_msg = 'No text could be extracted from the PDF (file might be image-based or encrypted)'
            tool_context.state['processing_history'].append({
                'action': 'text_extraction',
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': error_msg,
                'file_path': pdf_path,
                'pages_processed': pages_processed
            })
            return {
                'status': 'error',
                'message': error_msg,
                'document_path': pdf_path
            }
            
        # Update state with extracted text
        if 'text' not in tool_context.state['extracted_data']:
            tool_context.state['extracted_data']['text'] = []
            
        text_entry = {
            'source': os.path.basename(pdf_path),
            'content': text,
            'extracted_at': datetime.now().isoformat(),
            'pages_processed': pages_processed,
            'total_pages': len(doc)
        }
        tool_context.state['extracted_data']['text'].append(text_entry)
        
        # Update document status
        tool_context.state['document_status']['text_extracted'] = True
        tool_context.state['document_status']['pages_processed'] = pages_processed
        tool_context.state['document_status']['total_pages'] = len(doc)
        
        # Log successful extraction
        tool_context.state['processing_history'].append({
            'action': 'text_extraction',
            'timestamp': datetime.now().isoformat(),
            'status': 'success',
            'pages_processed': pages_processed,
            'total_pages': len(doc),
            'text_length': len(text),
            'file_path': pdf_path
        })
        
        return {
            'status': 'success' if pages_processed > 0 else 'partial_success',
            'text': text,
            'pages_processed': pages_processed,
            'total_pages': len(doc),
            'document_path': pdf_path,
            'warnings': 'Some pages could not be processed' if pages_processed < len(doc) else None
        }
        
    except Exception as e:
        # Log any unexpected errors
        error_msg = f'Unexpected error during PDF text extraction: {str(e)}'
        tool_context.state['processing_history'].append({
            'action': 'text_extraction',
            'timestamp': datetime.now().isoformat(),
            'status': 'error',
            'error': error_msg,
            'file_path': pdf_path,
            'exception_type': type(e).__name__
        })
        
        return {
            'status': 'error',
            'message': error_msg,
            'document_path': pdf_path,
            'exception_type': type(e).__name__
        }
        
    finally:
        # Ensure the document is always closed
        if doc:
            try:
                doc.close()
            except Exception as e:
                # Log but don't fail if there's an error closing
                tool_context.state['processing_history'].append({
                    'action': 'document_cleanup',
                    'timestamp': datetime.now().isoformat(),
                    'status': 'warning',
                    'message': f'Error closing PDF document: {str(e)}',
                    'file_path': pdf_path
                })

def extract_citations(tool_context: ToolContext, text: str) -> Dict[str, Any]:
    """
    Extract citations and references from text.
    
    Args:
        tool_context: The context containing state and other utilities
        text: Text content to extract citations from
        
    Returns:
        Dict containing extracted citations and status
    """
    # Initialize state if needed
    if 'extracted_data' not in tool_context.state:
        tool_context.state['extracted_data'] = {}
    if 'document_status' not in tool_context.state:
        tool_context.state['document_status'] = {}
    if 'processing_history' not in tool_context.state:
        tool_context.state['processing_history'] = []
    
    try:
        # Simple citation extraction (can be enhanced with more sophisticated methods)
        # This is a basic implementation - consider using a citation parser library for production
        citation_pattern = r'\[(\d+)\]|(\([A-Za-z]+\s*\d{4}\))'
        citations = re.findall(citation_pattern, text)
        
        # Process citations
        unique_citations = list(set([c[0] or c[1] for c in citations if c[0] or c[1]]))
        citations_list = [{'id': i+1, 'reference': ref} for i, ref in enumerate(unique_citations)]
        
        # Update state
        tool_context.state['extracted_data']['citations'] = citations_list
        tool_context.state['document_status']['citations_extracted'] = True
        
        # Log this action
        tool_context.state['processing_history'].append({
            'action': 'citation_extraction',
            'timestamp': datetime.now().isoformat(),
            'status': 'success',
            'citation_count': len(citations_list)
        })
        
        return {
            'status': 'success',
            'citation_count': len(citations_list),
            'citations': citations_list[:10]  # Return first 10 citations as example
        }
        
    except Exception as e:
        # Log the error
        tool_context.state['processing_history'].append({
            'action': 'citation_extraction',
            'timestamp': datetime.now().isoformat(),
            'status': 'error',
            'error': str(e)
        })
        
        return {
            'status': 'error',
            'error': str(e)
        }

def extract_figures_tables(tool_context: ToolContext, pdf_path: str) -> Dict[str, Any]:
    """
    Extract figures and tables from a PDF document.
    
    Args:
        tool_context: The context containing state and other utilities
        pdf_path: Path to the PDF file
        
    Returns:
        Dict containing extracted figures and tables with status
    """
    # Initialize state if needed
    if 'extracted_data' not in tool_context.state:
        tool_context.state['extracted_data'] = {}
    if 'document_status' not in tool_context.state:
        tool_context.state['document_status'] = {}
    if 'processing_history' not in tool_context.state:
        tool_context.state['processing_history'] = []
    
    try:
        # Open the PDF file
        doc = fitz.open(pdf_path)
        
        # Initialize lists to store figures and tables
        figures = []
        tables = []
        
        # Extract images and tables from each page
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Extract images (figures)
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                figures.append({
                    'page': page_num + 1,
                    'index': img_index,
                    'position': 'top' if page_num < len(doc) / 2 else 'bottom',
                    'caption': f"Figure {len(figures) + 1}"  # Placeholder caption
                })
            
            # Extract tables (this is a simplified example)
            # In production, you might want to use a dedicated table extraction library
            tables_on_page = page.find_tables()
            for table_index, table in enumerate(tables_on_page.tables):
                tables.append({
                    'page': page_num + 1,
                    'index': table_index,
                    'position': 'top' if page_num < len(doc) / 2 else 'bottom',
                    'shape': f"{table.rows}x{table.cols}",
                    'caption': f"Table {len(tables) + 1}"  # Placeholder caption
                })
        
        # Update state
        tool_context.state['extracted_data']['figures'] = figures
        tool_context.state['extracted_data']['tables'] = tables
        tool_context.state['document_status']['figures_tables_extracted'] = True
        
        # Log this action
        tool_context.state['processing_history'].append({
            'action': 'figures_tables_extraction',
            'timestamp': datetime.now().isoformat(),
            'status': 'success',
            'figure_count': len(figures),
            'table_count': len(tables)
        })
        
        return {
            'status': 'success',
            'figure_count': len(figures),
            'table_count': len(tables)
        }
        
    except Exception as e:
        # Log the error
        tool_context.state['processing_history'].append({
            'action': 'figures_tables_extraction',
            'timestamp': datetime.now().isoformat(),
            'status': 'error',
            'error': str(e)
        })
        
        return {
            'status': 'error',
            'error': str(e)
        }

def validate_extraction_quality(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Validate the quality of the extracted content.
    
    Args:
        tool_context: The context containing state and other utilities
        
    Returns:
        Dict containing validation results and status
    """
    # Initialize state if needed
    if 'document_status' not in tool_context.state:
        tool_context.state['document_status'] = {}
    if 'processing_history' not in tool_context.state:
        tool_context.state['processing_history'] = []
    
    try:
        # Check which extraction steps were completed
        status = tool_context.state.get('document_status', {})
        extractions = tool_context.state.get('extracted_data', {})
        
        # Basic validation checks
        validation_results = {
            'metadata_valid': bool(extractions.get('metadata', {}).get('title')),
            'text_extracted': bool(extractions.get('full_text')),
            'citations_found': bool(extractions.get('citations')),
            'figures_tables_found': bool(extractions.get('figures') or extractions.get('tables')),
            'all_required_steps_completed': all(status.get(key, False) for key in [
                'metadata_extracted', 'text_extracted', 'citations_extracted', 'figures_tables_extracted'
            ])
        }
        
        # Calculate an overall quality score (simple average of validation checks)
        quality_score = sum(1 for v in validation_results.values() if v) / len(validation_results)
        
        # Update state
        tool_context.state['document_status']['validation_complete'] = True
        tool_context.state['extracted_data']['validation_results'] = {
            'quality_score': quality_score,
            'details': validation_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Log this action
        tool_context.state['processing_history'].append({
            'action': 'quality_validation',
            'timestamp': datetime.now().isoformat(),
            'status': 'success',
            'quality_score': quality_score,
            'validation_details': validation_results
        })
        
        return {
            'status': 'success',
            'quality_score': quality_score,
            'validation_details': validation_results
        }
        
    except Exception as e:
        # Log the error
        tool_context.state['processing_history'].append({
            'action': 'quality_validation',
            'timestamp': datetime.now().isoformat(),
            'status': 'error',
            'error': str(e)
        })
        
        return {
            'status': 'error',
            'error': str(e)
        }
import os
from pathlib import Path

# Get the absolute path to the project root (assuming tools.py is in a subdirectory)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
UPLOAD_FOLDER = str(PROJECT_ROOT / "data" / "uploads")
PROCESSED_FOLDER = str(PROJECT_ROOT / "data" / "processed")

def ensure_upload_dirs():
    """Ensure upload and processed directories exist."""
    try:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(PROCESSED_FOLDER, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directories: {e}")
        return False



def process_document(
    tool_context: ToolContext,
    document_path: Optional[str] = None,
    document_content: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Process a document through the ingestion pipeline with enhanced error handling and session management.

    Args:
        tool_context: The tool context containing state and utilities
        document_path: Path to the document to process (optional if document_content is provided)
        document_content: Content of the document as string or bytes (optional if document_path is provided)
        options: Additional processing options including:
            - extract_metadata: Whether to extract metadata (default: True)
            - extract_text: Whether to extract text content (default: True)
            - extract_citations: Whether to extract citations (default: True)
            - extract_figures_tables: Whether to extract figures and tables (default: True)
            - session_id: Optional session ID for tracking

    Returns:
        Dict containing processing results including:
            - status: 'success', 'partial_success', or 'error'
            - document_id: ID of the processed document
            - message: Status message
            - document: Document metadata and processing status
            - session_id: The session ID used for this processing
    """
    # Initialize session and timing
    start_time = datetime.utcnow()
    session_id = None
    temp_file_path = None
    doc_id = None
    
    # Helper function to clean up resources
    def cleanup():
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass  # Best effort cleanup
    
    try:
        # Initialize state if needed
        if 'documents' not in tool_context.state:
            tool_context.state['documents'] = {}
        if 'sessions' not in tool_context.state:
            tool_context.state['sessions'] = {}
            
        # Get or create session
        session_id = options.get('session_id') if options else None
        if not session_id:
            session_id = f'sess_{int(datetime.utcnow().timestamp())}_{len(tool_context.state.get("sessions", {}))}'
        
        # Initialize session data if it doesn't exist
        if session_id not in tool_context.state['sessions']:
            tool_context.state['sessions'][session_id] = {
                'created_at': datetime.utcnow().isoformat(),
                'documents_processed': 0,
                'documents_failed': 0,
                'last_activity': datetime.utcnow().isoformat(),
                'status': 'active'
            }
        
        # Update session activity
        tool_context.state['sessions'][session_id]['last_activity'] = datetime.utcnow().isoformat()
        
        # Set default options if not provided
        default_options = {
            'extract_metadata': True,
            'extract_text': True,
            'extract_citations': True,
            'extract_figures_tables': True,
            'session_id': session_id
        }
        options = {**default_options, **(options or {})}
        
        # Generate a unique document ID
        doc_id = f'doc_{int(datetime.utcnow().timestamp())}_{len(tool_context.state["documents"])}'
        
        # Handle document content if provided instead of file path
        doc_name = None
        file_info = {}
        
        try:
            if document_content is not None and not document_path:
                # Create a temporary file in the UPLOAD_FOLDER
                import uuid
                
                # Ensure upload directories exist
                if not ensure_upload_dirs():
                    raise Exception('Failed to create upload directories')
                
                # Generate a unique filename with original extension if available
                file_ext = '.bin'  # Default extension for binary data
                if hasattr(tool_context, 'filename') and '.' in tool_context.filename:
                    file_ext = os.path.splitext(tool_context.filename)[1]
                elif isinstance(document_content, str) and len(document_content) > 1000:
                    # Try to determine file type from content
                    if document_content.startswith('%PDF'):
                        file_ext = '.pdf'
                    elif document_content.startswith('\\x50\\x4B\\x03\\x04'):  # ZIP header
                        file_ext = '.docx'
                
                # Create a temporary file with a unique name
                temp_filename = f"{doc_id}{file_ext}"
                temp_file_path = os.path.join(UPLOAD_FOLDER, temp_filename)
                
                # Write the content to the file
                with open(temp_file_path, 'wb') as f:
                    if isinstance(document_content, str):
                        f.write(document_content.encode('utf-8'))
                    else:
                        f.write(document_content)
                
                document_path = temp_file_path
                doc_name = temp_filename
                
            elif document_path:
                # Use the provided document path
                doc_name = os.path.basename(document_path)
                
                # Make sure the path is absolute
                if not os.path.isabs(document_path):
                    document_path = os.path.abspath(document_path)
                
                # Verify the file exists and is accessible
                if not os.path.exists(document_path):
                    raise FileNotFoundError(f'Document not found: {document_path}')
                if not os.path.isfile(document_path):
                    raise ValueError(f'Path is not a file: {document_path}')
                if not os.access(document_path, os.R_OK):
                    raise PermissionError(f'Cannot read file (permission denied): {document_path}')
                
                # Get file info
                file_info = {
                    'size': os.path.getsize(document_path),
                    'last_modified': datetime.fromtimestamp(
                        os.path.getmtime(document_path)
                    ).isoformat(),
                    'file_type': os.path.splitext(document_path)[1].lower().lstrip('.')
                }
            else:
                raise ValueError('Either document_path or document_content must be provided')
            
            # Initialize document state
            tool_context.state['documents'][doc_id] = {
                'id': doc_id,
                'name': doc_name,
                'path': document_path,
                'status': 'processing',
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat(),
                'metadata': {},
                'processing_steps': [],
                'file_info': file_info,
                'session_id': session_id,
                'options': options
            }
            
            # Process the document based on options
            result = {
                'status': 'success',
                'document_id': doc_id,
                'document_name': doc_name,
                'processing_steps': [],
                'session_id': session_id
            }
            
            # Extract text if requested
            if options.get('extract_text', True):
                try:
                    text_result = extract_text_from_pdf(tool_context, document_path)
                    tool_context.state['documents'][doc_id]['text_extraction'] = text_result
                    tool_context.state['documents'][doc_id]['processing_steps'].append({
                        'step': 'text_extraction',
                        'status': text_result.get('status', 'unknown'),
                        'timestamp': datetime.utcnow().isoformat()
                    })
                    result['processing_steps'].append({
                        'step': 'text_extraction',
                        'status': text_result.get('status', 'unknown'),
                        'pages_processed': text_result.get('pages_processed', 0),
                        'total_pages': text_result.get('total_pages', 0)
                    })
                except Exception as e:
                    error_msg = f'Error extracting text: {str(e)}'
                    tool_context.state['documents'][doc_id]['processing_steps'].append({
                        'step': 'text_extraction',
                        'status': 'error',
                        'error': error_msg,
                        'timestamp': datetime.utcnow().isoformat()
                    })
                    result['processing_steps'].append({
                        'step': 'text_extraction',
                        'status': 'error',
                        'error': error_msg
                    })
                    result['status'] = 'partial_success'
            
            # Update document status
            tool_context.state['documents'][doc_id]['status'] = 'processed'
            tool_context.state['documents'][doc_id]['updated_at'] = datetime.utcnow().isoformat()
            
            # Update session statistics
            tool_context.state['sessions'][session_id]['documents_processed'] += 1
            tool_context.state['sessions'][session_id]['last_activity'] = datetime.utcnow().isoformat()
            
            return result
            
        except Exception as e:
            # Update session statistics
            if 'sessions' in tool_context.state and session_id in tool_context.state['sessions']:
                tool_context.state['sessions'][session_id]['documents_failed'] += 1
                tool_context.state['sessions'][session_id]['last_activity'] = datetime.utcnow().isoformat()
            
            # Update document status if it was created
            if doc_id and 'documents' in tool_context.state and doc_id in tool_context.state['documents']:
                tool_context.state['documents'][doc_id].update({
                    'status': 'error',
                    'error': str(e),
                    'updated_at': datetime.utcnow().isoformat()
                })
            
            raise  # Re-raise the exception to be caught by the outer try-except
            
    except Exception as e:
        error_info = {
            'status': 'error',
            'message': f'Error processing document: {str(e)}',
            'exception_type': type(e).__name__,
            'document_path': document_path,
            'document_id': doc_id,
            'session_id': session_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Add traceback if available
        import traceback
        error_info['traceback'] = traceback.format_exc()
        
        return error_info
    
    finally:
        # Clean up temporary files
        cleanup()