# scholar_verse_agents.py - Comprehensive agent implementation for ScholarVerse
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import types

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)





import os
import json
import logging
from typing import Dict, List, Optional, Union,Any
from pathlib import Path
import PyPDF2
from datetime import datetime
from dotenv import load_dotenv
import asyncio
load_dotenv()
# Google ADK imports
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.tools import FunctionTool,ToolContext
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService
from google.genai import types
from google.genai.types import Part, Blob
from google.adk.agents.invocation_context import InvocationContext

from sklearn.feature_extraction.text import TfidfVectorizer

# Set up logging
import os
from logging.handlers import RotatingFileHandler

# Create logs directory if it doesn't exist
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'scholarverse.log')

# Configure root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# File handler with rotation (10MB per file, keep 5 backups)
file_handler = RotatingFileHandler(
    log_file,
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setFormatter(formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Add handlers to root logger
logger.handlers = [file_handler, console_handler]

# Module logger
logger = logging.getLogger(__name__)
logger.info(f"Logging to file: {log_file}")



#custom model setup
model=LiteLlm(api_key=os.getenv("OPENROUTER_API_KEY"),model="openrouter/anthropic/claude-3.5-sonnet")
model="gemini-2.0-flash"







# ----------------------
# Tool Implementations
# ----------------------

# PDF Processing Tools
async def extract_text_from_pdf(tool_context:ToolContext,file_path: str) -> str:
    """
    Extract text content from a PDF file and store it as an artifact.

    Args:
        tool_context (ToolContext): The tool context containing state and artifacts.
        file_path (str): Path to the PDF file to extract text from.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - If successful: {'status': 'success', 'message': str, 'artifact_name': str}
            - If error: {'status': 'error', 'message': str}
    """
    try:
        context = tool_context  
        # Get the artifact if it exists
        artifact_name = os.path.basename(file_path)
        try:
            artifact = context.load_artifact(artifact_name)
            pdf_bytes = artifact.inline_data.data

            # Create a temporary file to work with PyPDF2
            temp_path = f"temp_{artifact_name}"
            with open(temp_path, "wb") as f:
                f.write(pdf_bytes)
                
            # Extract text using PyPDF2
            with open(temp_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                    
            # Clean up temp file
            os.remove(temp_path)
            
            # Save extracted text as a new artifact
            text_artifact_name = f"{artifact_name}_text"
            context.save_artifact(filename=text_artifact_name, artifact=types.Part(inline_data=types.Blob(mime_type="text/plain",data=text.encode('utf-8'))))
            
            return f"Successfully extracted text from {artifact_name}. Saved as artifact: {text_artifact_name}"
        except Exception as e:
            # If artifact doesn't exist, try to read from file path directly
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                
                # Save PDF as artifact first
                with open(file_path, "rb") as f:
                    pdf_bytes = f.read()
                    context.save_artifact(filename=artifact_name, artifact=types.Part(inline_data=types.Blob(mime_type="application/pdf",data=pdf_bytes)))
                
                # Save extracted text as artifact
                text_artifact_name = f"{artifact_name}_text"
                context.save_artifact(filename=text_artifact_name, artifact=types.Part(inline_data=types.Blob(mime_type="text/plain",data=text.encode('utf-8'))))
                
                return f"Successfully extracted text from {file_path}. Saved as artifact: {text_artifact_name}"
            else:
                return f"Error: File {file_path} not found"
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return f"Error extracting text: {str(e)}"

async def extract_metadata_from_pdf(tool_context:ToolContext,file_path: str) -> Dict:
    """
    Extract metadata from a PDF file and store it as an artifact.

    Args:
        tool_context (ToolContext): The tool context containing state and artifacts.
        file_path (str): Path to the PDF file to extract metadata from.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - If successful: {
                'status': 'success',
                'metadata': Dict[str, Any],  # Extracted metadata
                'artifact_name': str
              }
            - If error: {'status': 'error', 'message': str}
    """
    try:
        context=tool_context        # Get the artifact if it exists
        artifact_name = os.path.basename(file_path)
        try:
            artifact = context.load_artifact(artifact_name)
            pdf_bytes = artifact.inline_data.data
            
            # Create a temporary file to work with PyPDF2
            temp_path = f"temp_{artifact_name}"
            with open(temp_path, "wb") as f:
                f.write(pdf_bytes)
                
            # Extract metadata using PyPDF2
            with open(temp_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                metadata = reader.metadata
                info = {k: str(v) for k, v in metadata.items()} if metadata else {}
                
                # Add basic info
                info["num_pages"] = len(reader.pages)
                info["filename"] = artifact_name
                
            # Clean up temp file
            os.remove(temp_path)
            
            # Save metadata as artifact
            metadata_artifact_name = f"{artifact_name}_metadata"
            context.save_artifact(filename=metadata_artifact_name, artifact=types.Part(inline_data=types.Blob(mime_type="application/json",data=json.dumps(info).encode('utf-8'))))
            
            return info
        except Exception as e:
            # If artifact doesn't exist, try to read from file path directly
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    metadata = reader.metadata
                    info = {k: str(v) for k, v in metadata.items()} if metadata else {}
                    
                    # Add basic info
                    info["num_pages"] = len(reader.pages)
                    info["filename"] = os.path.basename(file_path)
                
                # Save PDF as artifact first if not already saved
                with open(file_path, "rb") as f:
                    pdf_bytes = f.read()
                    context.save_artifact(filename=artifact_name, artifact=types.Part(inline_data=types.Blob(mime_type="application/pdf",data=pdf_bytes)))
                
                # Save metadata as artifact
                metadata_artifact_name = f"{artifact_name}_metadata"
                context.save_artifact(filename=metadata_artifact_name, artifact=types.Part(inline_data=types.Blob(mime_type="application/json",data=json.dumps(info).encode('utf-8'))))
                
                return info
            else:
                return {"error": f"File {file_path} not found"}
    except Exception as e:
        logger.error(f"Error extracting metadata from PDF: {str(e)}")
        return {"error": str(e)}

async def process_pdf_directory(tool_context: ToolContext, directory_path: str) -> Dict:
    """
    Process all PDF files in a directory, extract text, and store both PDFs and text as artifacts.
    Creates a pdf_directory_index artifact with metadata about all processed documents.

    Args:
        tool_context (ToolContext): The tool context containing state and artifacts.
        directory_path (str): Path to the directory containing PDF files.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - If successful: {
                'status': 'success',
                'message': str,
                'processed_files': List[Dict],  # List of processed file info
                'index_artifact': str,  # Name of the directory index artifact
                'text_artifacts': List[str]  # Names of created text artifacts
              }
            - If error: {'status': 'error', 'message': str}
    """
    try:
        if not os.path.exists(directory_path):
            return {"status": "error", "message": f"Directory {directory_path} not found"}
        
        pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
        if not pdf_files:
            return {"status": "error", "message": f"No PDF files found in {directory_path}"}
        
        results = {}
        text_artifacts = []
        doc_no = 1
        
        for pdf_file in pdf_files:
            try:
                file_path = os.path.join(directory_path, pdf_file)
                base_name = os.path.splitext(pdf_file)[0]
                
                # Read PDF bytes
                with open(file_path, "rb") as f:
                    pdf_bytes = f.read()
                
                # Save PDF as artifact using Part.from_data
                pdf_artifact = types.Part(
                    inline_data=types.Blob(
                        mime_type="application/pdf",
                        data=pdf_bytes
                    )
                )
                tool_context.save_artifact(filename=pdf_file, artifact=pdf_artifact)
                
                # Extract text from PDF
                text = ""
                try:
                    with open(file_path, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        text = "\n\n".join(page.extract_text() for page in reader.pages if page.extract_text())
                except Exception as e:
                    error_msg = f"Error extracting text from {pdf_file}: {str(e)}"
                    logger.warning(error_msg)
                    text = f"[Error extracting text: {str(e)}]"
                
                # Save text as artifact using Part.from_data
                text_artifact_name = f"{base_name}_text"
                text_data = json.dumps({
                    "text": text,
                    "source_pdf": pdf_file,
                    "extracted_at": datetime.now().isoformat()
                }).encode('utf-8')
                
                text_artifact = types.Part(
                    inline_data=types.Blob(
                        mime_type="application/json",
                        data=text_data
                    )
                )
                tool_context.save_artifact(text_artifact_name, text_artifact)
                
                text_artifacts.append(text_artifact_name)
                
                # Add to results
                results[pdf_file] = {
                    "doc_id": doc_no,
                    "path": file_path,
                    "artifact_name": pdf_file,
                    "text_artifact": text_artifact_name,
                    "num_pages": len(PyPDF2.PdfReader(open(file_path, "rb")).pages) if os.path.exists(file_path) else 0,
                    "processed": True,
                    "processed_at": datetime.now().isoformat()
                }
                
                doc_no += 1
                
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {str(e)}")
                results[pdf_file] = {
                    "error": str(e),
                    "processed": False
                }
        
        # Save directory index as artifact with metadata
        index_data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "documents": results,
            "text_artifacts": text_artifacts
        }
        
        index_artifact = types.Part(
            inline_data=types.Blob(
                mime_type="application/json",
                data=json.dumps(index_data, indent=2).encode('utf-8')
            )
        )
        tool_context.save_artifact(filename="pdf_directory_index", artifact=index_artifact)
        
        # Update session state
        if 'documents' not in tool_context.state:
            tool_context.state['documents'] = {}
        
        for pdf_file, data in results.items():
            tool_context.state['documents'][pdf_file] = {
                "path": data.get("path", ""),
                "artifact_name": data.get("artifact_name", ""),
                "text_artifact": data.get("text_artifact", ""),
                "processed": data.get("processed", False),
                "created_at": datetime.now().isoformat()
            }
        
        return {
            "status": "success",
            "message": f"Successfully processed {len([f for f in results.values() if f.get('processed', False)])} PDF files from {directory_path}",
            "processed_files": [{"file": k, "status": "success" if v.get("processed", False) else "failed"} 
                              for k, v in results.items()],
            "index_artifact": "pdf_directory_index",
            "text_artifacts": text_artifacts
        }
        
    except Exception as e:
        error_msg = f"Error processing PDF directory: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"status": "error", "message": error_msg}

# Utility Functions
def save_analysis_output(task_name: str, data: dict) -> str:
    """
    Save analysis output as an artifact and to a file in the outputs directory.
    
    Args:
        task_name (str): Name of the analysis task
        data (dict): Data to save
        
    Returns:
        str: Path to the saved file
    """
    try:
        # Create outputs directory if it doesn't exist
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{task_name}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Convert data to JSON bytes
        json_data = json.dumps(data, indent=2, ensure_ascii=False).encode('utf-8')
        
        # Save to file
        with open(filepath, 'wb') as f:
            f.write(json_data)
            
        # Also save as an artifact
        artifact_name = f"analysis_output_{task_name}_{timestamp}"
        output_artifact = Part(
            inline_data=Blob(
                mime_type="application/json",
                data=json_data
            )
        )
        
        # Get the tool context from the current context or use a global one
        try:
            from google.adk.agents.callback_context import get_current_context
            context = get_current_context()
            if context:
                context.save_artifact(filename=artifact_name, artifact=output_artifact)
                logger.info(f"Saved {task_name} output as artifact: {artifact_name}")
        except Exception as e:
            logger.warning(f"Could not save analysis output as artifact: {str(e)}")
        
        logger.info(f"Saved {task_name} output to {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Error saving analysis output: {str(e)}", exc_info=True)
        return ""

# Cross Paper Analysis Tools

async def _process_documents_for_analysis(
    tool_context: ToolContext,
    document_ids: str,
    task_name: str,
    analysis_type: str
) -> Dict:
    """Helper function to process 10-20 documents for topic analysis using TF-IDF and LLM."""
    try:
        logger.info(f"Starting document processing for task: {task_name}")
        
        # Parse the comma-separated string and clean up
        doc_ids = [id_str.strip() for id_str in document_ids.split(',') if id_str.strip()]
        logger.info(f"Processing document IDs: {doc_ids}")
        
        if len(doc_ids) < 2:
            error_msg = "Need at least two document IDs to compare"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
        
        if len(doc_ids) > 20:
            error_msg = "Exceeds maximum of 20 documents for topic analysis"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}

        # Load the directory index
        try:
            logger.info("Attempting to load pdf_directory_index artifact")
            artifact = tool_context.load_artifact("pdf_directory_index")
            
            if not artifact or not hasattr(artifact, 'inline_data') or not artifact.inline_data:
                error_msg = "Failed to load pdf_directory_index: invalid artifact format"
                logger.error(error_msg)
                return {"status": "error", "message": error_msg}
            
            raw_data = artifact.inline_data.data
            if isinstance(raw_data, bytes):
                try:
                    raw_data = raw_data.decode('utf-8')
                except UnicodeDecodeError as e:
                    error_msg = f"Failed to decode pdf_directory_index: {str(e)}"
                    logger.error(error_msg)
                    return {"status": "error", "message": error_msg}
            
            try:
                index_data = json.loads(raw_data)
                logger.info(f"Successfully loaded pdf_directory_index with {len(index_data.get('documents', {}))} documents")
            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse pdf_directory_index JSON: {str(e)}"
                logger.error(error_msg)
                return {"status": "error", "message": error_msg}
            
        except Exception as e:
            error_msg = f"Could not load pdf_directory_index: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"status": "error", "message": error_msg}

        # Map document IDs to artifact names
        artifact_names = []
        documents = index_data.get('documents', {})
        
        for doc_id in doc_ids:
            found = False
            for artifact_name, data in documents.items():
                if not isinstance(data, dict):
                    logger.warning(f"Unexpected data format in index for {artifact_name}: {data}")
                    continue
                
                if (str(data.get('doc_id')) == doc_id or 
                    artifact_name == doc_id or 
                    data.get('text_artifact', '').startswith(doc_id)):
                    
                    text_artifact = data.get('text_artifact', f"{os.path.splitext(artifact_name)[0]}_text")
                    artifact_names.append({
                        'pdf_artifact': artifact_name,
                        'text_artifact': text_artifact,
                        'doc_id': data.get('doc_id')
                    })
                    found = True
                    logger.info(f"Mapped document ID {doc_id} to artifact: {artifact_name} (text: {text_artifact})")
                    break
                    
            if not found:
                available_docs = [{
                    'doc_id': d.get('doc_id'),
                    'artifact': k,
                    'text_artifact': d.get('text_artifact', f"{os.path.splitext(k)[0]}_text")
                } for k, d in documents.items() if isinstance(d, dict)]
                error_msg = (
                    f"Document with ID {doc_id} not found in index. "
                    f"Available documents: {json.dumps(available_docs, indent=2)}"
                )
                logger.error(error_msg)
                return {"status": "error", "message": error_msg}

        # Process documents and load their text content
        documents_text = {}
        for doc_info in artifact_names:
            artifact_name = doc_info['pdf_artifact']
            text_artifact_name = doc_info['text_artifact']
            
            try:
                artifact = tool_context.load_artifact(text_artifact_name)
                if not artifact or not hasattr(artifact, 'inline_data'):
                    logger.warning(f"Text artifact {text_artifact_name} is invalid or missing inline_data")
                    documents_text[artifact_name] = "[Error: Invalid text artifact]"
                    continue
                
                content = artifact.inline_data.data
                if isinstance(content, bytes):
                    try:
                        content = content.decode('utf-8')
                        if content.strip().startswith('{'):
                            try:
                                text_data = json.loads(content)
                                if isinstance(text_data, dict) and 'text' in text_data:
                                    content = text_data['text']
                            except json.JSONDecodeError:
                                pass
                    except UnicodeDecodeError:
                        content = f"[Error: Cannot decode text artifact {text_artifact_name}]"
                
                documents_text[artifact_name] = content
                logger.info(f"Successfully loaded text for {artifact_name} from {text_artifact_name}")
                
            except Exception as e:
                logger.warning(f"Could not load text artifact {text_artifact_name}: {str(e)}")
                try:
                    pdf_artifact = tool_context.load_artifact(artifact_name)
                    pdf_bytes = pdf_artifact.inline_data.data
                    with open("temp_pdf.pdf", "wb") as f:
                        f.write(pdf_bytes)
                    with open("temp_pdf.pdf", "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        text = "\n\n".join(page.extract_text() for page in reader.pages if page.extract_text())
                    os.remove("temp_pdf.pdf")
                    
                    text_artifact_data = json.dumps({
                        "text": text,
                        "source_pdf": artifact_name,
                        "extracted_at": datetime.now().isoformat(),
                        "extraction_method": "fallback_pdf_extraction"
                    }).encode('utf-8')
                    
                    text_artifact = Part(
                        inline_data=Blob(
                            mime_type="application/json",
                            data=text_artifact_data
                        )
                    )
                    tool_context.save_artifact(filename=text_artifact_name, artifact=text_artifact)
                    documents_text[artifact_name] = text
                    logger.info(f"Successfully extracted and saved text for {artifact_name}")
                    
                except Exception as inner_e:
                    error_msg = f"Failed to process {artifact_name}: {str(inner_e)}"
                    logger.error(error_msg, exc_info=True)
                    documents_text[artifact_name] = f"[Error processing document: {str(inner_e)}]"

        # Extract keywords using TF-IDF
        valid_texts = [text for text in documents_text.values() if not text.startswith("[Error")]
        valid_doc_names = [name for name, text in documents_text.items() if not text.startswith("[Error")]
        
        if len(valid_texts) < 2:
            error_msg = "Fewer than two valid documents available for topic analysis"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
        
        try:
            # Calculate min_df and max_df to ensure max_df is always greater than min_df
            n_docs = len(valid_texts)
            min_docs = min(2, max(1, int(n_docs * 0.2)))  # At least 1 doc, at most 2 or 20% of docs
            max_docs = max(min_docs + 1, int(n_docs * 0.8))  # At least min_docs + 1, at most 80% of docs
            
            vectorizer = TfidfVectorizer(
                max_features=50,  # Limit to top 50 keywords
                stop_words='english',
                max_df=0.8,  # Ignore terms in >80% of documents
                min_df=min_docs  # Terms must appear in at least min_docs documents
            )
            tfidf_matrix = vectorizer.fit_transform(valid_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Extract top keywords per document
            doc_keywords = {}
            for i, doc_name in enumerate(valid_doc_names):
                scores = tfidf_matrix[i].toarray()[0]
                top_indices = scores.argsort()[-10:][::-1]  # Top 10 keywords
                doc_keywords[doc_name] = [feature_names[idx] for idx in top_indices if scores[idx] > 0]
            
            # Save keywords as an artifact
            keywords_artifact_name = f"keywords_{analysis_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            keywords_data = {
                "documents": doc_keywords,
                "created_at": datetime.now().isoformat()
            }
            tool_context.save_artifact(
                filename=keywords_artifact_name,
                artifact=Part(inline_data=Blob(mime_type="application/json", data=json.dumps(keywords_data).encode('utf-8')))
            )
            logger.info(f"Saved keywords as artifact: {keywords_artifact_name}")
            
        except Exception as e:
            logger.error(f"Error during TF-IDF processing: {str(e)}")
            return {"status": "error", "message": f"TF-IDF processing failed: {str(e)}"}

        # Batch processing for LLM
        batch_size = 5
        all_topics = []
        for i in range(0, len(valid_doc_names), batch_size):
            batch_docs = valid_doc_names[i:i + batch_size]
            batch_keywords = {doc: doc_keywords[doc] for doc in batch_docs}
            
            # Prepare prompt for LLM
            prompt = (
                "You are an expert in research paper analysis. Below are keywords extracted from multiple research papers. "
                "Your task is to identify up to 5 common topics or themes across these papers based on the keywords. "
                "Each topic should be specific, relevant, and avoid generic terms like 'research' or 'study'. "
                "Return a JSON object with a 'topics' key containing a list of dictionaries, "
                "each with 'topic' and 'description' keys.\n\n"
            )
            
            for doc_name, keywords in batch_keywords.items():
                prompt += f"### Paper: {doc_name}\nKeywords: {', '.join(keywords)}\n\n"
            
            prompt += (
                "\nBased on the keywords, identify and describe up to 5 common topics. "
                "Return the result in JSON format:\n"
                "```json\n"
                "{\"topics\": [{\"topic\": \"topic_name\", \"description\": \"description\"}, ...]}\n"
                "```"
            )
            
            # Call the LLM
            try:
                llm_response = await tool_context.model.generate_content_async(
                    contents=[types.Content(role="user", parts=[types.Part(text=prompt)])]
                )
                
                if llm_response and llm_response.parts:
                    try:
                        response_text = llm_response.parts[0].text
                        if response_text.startswith("```json"):
                            response_text = response_text.strip("```json\n").strip("```")
                        topics_data = json.loads(response_text)
                        all_topics.extend(topics_data.get("topics", []))
                        logger.info(f"Successfully extracted topics for batch {i//batch_size + 1}")
                    except (json.JSONDecodeError, AttributeError) as e:
                        logger.error(f"Error parsing LLM response for batch {i//batch_size + 1}: {str(e)}")
                        continue
                else:
                    logger.error(f"No valid response from LLM for batch {i//batch_size + 1}")
                    continue
            except Exception as e:
                logger.error(f"LLM call failed for batch {i//batch_size + 1}: {str(e)}")
                continue
        
        # Deduplicate and limit topics
        unique_topics = []
        seen_topics = set()
        for topic in all_topics:
            topic_name = topic.get("topic", "").lower()
            if topic_name and topic_name not in seen_topics:
                unique_topics.append(topic)
                seen_topics.add(topic_name)
            if len(unique_topics) >= 5:
                break
        
        # Save analysis input as artifact
        analysis_input_name = f"{analysis_type}_input_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        result_data = {
            "task": task_name,
            "timestamp": datetime.now().isoformat(),
            "document_ids": [doc['doc_id'] for doc in artifact_names],
            "document_names": [doc['pdf_artifact'] for doc in artifact_names],
            "documents": {name: text[:500] + "..." if len(text) > 500 else text for name, text in documents_text.items()},
            "keywords": doc_keywords,
            "topics": unique_topics,
            "status": "success"
        }
        tool_context.save_artifact(
            filename=analysis_input_name,
            artifact=Part(inline_data=Blob(mime_type="application/json", data=json.dumps(result_data).encode('utf-8')))
        )
        
        # Save results to file
        output_file = save_analysis_output(f"{analysis_type}_analysis", result_data)
        
        # Prepare response
        response = {
            "status": "success",
            "message": f"Identified common topics across {len(valid_doc_names)} documents",
            "document_ids": [doc['doc_id'] for doc in artifact_names],
            "document_names": [doc['pdf_artifact'] for doc in artifact_names],
            "analysis_input_artifact": analysis_input_name,
            "output_file": output_file,
            "topics": unique_topics
        }
        
        logger.info(f"Analysis results saved to: {os.path.abspath(output_file)}" if output_file else "Failed to save analysis results to file")
        return response
        
    except Exception as e:
        logger.error(f"Error in {task_name}: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}
# async def identify_common_topics(tool_context: ToolContext, document_ids: str) -> Dict:
#     """
#     Identify common topics across multiple research papers.

#     Args:
#         tool_context (ToolContext): The tool context containing state and artifacts.
#         document_ids (str): Comma-separated string of document IDs to analyze (e.g., "1,2,3").

#     Returns:
#         Dict[str, Any]: A dictionary containing:
#             - If successful: {
#                 'status': 'success',
#                 'message': str,
#                 'document_ids': List[str],
#                 'analysis_input_artifact': str,
#                 'common_topics': List[str]  # List of identified topics
#               }
#             - If error: {'status': 'error', 'message': str}
#     """
#     return await _process_documents_for_analysis(
#         tool_context, 
#         document_ids, 
#         "identify_common_topics",
#         "topic_analysis"
#     )
async def identify_common_topics(tool_context: ToolContext, document_ids: str) -> Dict:
       """
       Identify common topics across multiple research papers.

       Args:
           tool_context (ToolContext): The tool context containing state and artifacts.
           document_ids (str): Comma-separated string of document IDs to analyze (e.g., "1,2,3").

       Returns:
           Dict[str, Any]: A dictionary containing:
               - If successful: {
                   'status': 'success',
                   'message': str,
                   'document_ids': List[str],
                   'analysis_input_artifact': str,
                   'topics': List[Dict[str, str]]  # List of topics with descriptions
                 }
               - If error: {'status': 'error', 'message': str}
       """
       return await _process_documents_for_analysis(
           tool_context, 
           document_ids, 
           "identify_common_topics",
           "topic_analysis"
       )
async def compare_methodologies(tool_context: ToolContext, document_ids: str) -> Dict:
    """
    Compare methodologies across multiple research papers.

    Args:
        tool_context (ToolContext): The tool context containing state and artifacts.
        document_ids (str): Comma-separated string of document IDs to compare (e.g., "1,2,3").

    Returns:
        Dict[str, Any]: A dictionary containing:
            - If successful: {
                'status': 'success',
                'message': str,
                'document_ids': List[str],
                'analysis_input_artifact': str,
                'methodology_comparison': Dict[str, Any]  # Comparison results
              }
            - If error: {'status': 'error', 'message': str}
    """
    return await _process_documents_for_analysis(
        tool_context, 
        document_ids, 
        "compare_methodologies",
        "methodology_analysis"
    )

async def compare_findings(tool_context: ToolContext, document_ids: str) -> Dict:
    """
    Compare findings across multiple research papers.

    Args:
        tool_context (ToolContext): The tool context containing state and artifacts.
        document_ids (str): Comma-separated string of document IDs to compare (e.g., "1,2,3").

    Returns:
        Dict[str, Any]: A dictionary containing:
            - If successful: {
                'status': 'success',
                'message': str,
                'document_ids': List[str],
                'analysis_input_artifact': str,
                'findings_comparison': Dict[str, Any]  # Comparison results
              }
            - If error: {'status': 'error', 'message': str}
    """
    return await _process_documents_for_analysis(
        tool_context, 
        document_ids, 
        "compare_findings",
        "findings_analysis"
    )

async def generate_cross_paper_analysis(tool_context:ToolContext, analysis_input_artifact: str) -> Dict:
    """Generate a comprehensive cross-paper analysis based on the prepared data."""
    try:
        # Retrieve the analysis input artifact
        artifact = tool_context.load_artifact(analysis_input_artifact)
        analysis_data = json.loads(artifact.inline_data.data.decode('utf-8'))
        
        task = analysis_data.get("task", "unknown")
        documents = analysis_data.get("documents", {})
        
        if not documents:
            return {"error": "No document data found in the analysis input"}
        
        # The actual analysis will be performed by the LLM
        # This function just organizes the data and passes it to the LLM
        
        # Save the analysis result when it comes back from the LLM
        # This would typically be done in a callback or after LLM processing
        
        # For now, return the metadata about what we're about to analyze
        return {
            "message": f"Ready to perform {task} analysis on {len(documents)} documents",
            "document_ids": list(documents.keys()),
            "task": task
        }
    except Exception as e:
        logger.error(f"Error generating cross-paper analysis: {str(e)}")
        return {"error": str(e)}

# Router Tools
async def analyze_document_complexity(tool_context:ToolContext, file_path: str) -> Dict:
    """
    Analyze the complexity of a document.

    Args:
        tool_context (ToolContext): The tool context containing state and artifacts.
        file_path (str): Path to the document to analyze.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - If successful: {
                'status': 'success',
                'filename': str,
                'num_pages': int,
                'avg_words_per_page': float,
                'complexity': str,  # 'low', 'medium', or 'high'
                'processing_strategy': str  # Recommended processing strategy
              }
            - If error: {'status': 'error', 'message': str}
    """
    try:
        # Try to get the artifact
        artifact_name = os.path.basename(file_path)
        try:
            artifact = tool_context.load_artifact(artifact_name)
            pdf_bytes = artifact.inline_data.data
            
            # Create a temporary file to work with PyPDF2
            temp_path = f"temp_{artifact_name}"
            with open(temp_path, "wb") as f:
                f.write(pdf_bytes)
                
            # Extract basic info using PyPDF2
            with open(temp_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                num_pages = len(reader.pages)
                
                # Sample some text to estimate complexity
                sample_text = ""
                sample_pages = min(5, num_pages)
                for i in range(sample_pages):
                    sample_text += reader.pages[i].extract_text() + "\n"
                
                word_count = len(sample_text.split())
                avg_words_per_page = word_count / sample_pages if sample_pages > 0 else 0
                
            # Clean up temp file
            os.remove(temp_path)
            
            # Determine complexity
            complexity = "low"
            if num_pages > 30 or avg_words_per_page > 500:
                complexity = "high"
            elif num_pages > 15 or avg_words_per_page > 300:
                complexity = "medium"
            
            return {
                "filename": artifact_name,
                "num_pages": num_pages,
                "avg_words_per_page": avg_words_per_page,
                "complexity": complexity
            }
        except Exception as e:
            # If artifact doesn't exist, try to read from file path directly
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    num_pages = len(reader.pages)
                    
                    # Sample some text to estimate complexity
                    sample_text = ""
                    sample_pages = min(5, num_pages)
                    for i in range(sample_pages):
                        sample_text += reader.pages[i].extract_text() + "\n"
                    
                    word_count = len(sample_text.split())
                    avg_words_per_page = word_count / sample_pages if sample_pages > 0 else 0
                
                # Determine complexity
                complexity = "low"
                if num_pages > 30 or avg_words_per_page > 500:
                    complexity = "high"
                elif num_pages > 15 or avg_words_per_page > 300:
                    complexity = "medium"
                
                return {
                    "filename": os.path.basename(file_path),
                    "num_pages": num_pages,
                    "avg_words_per_page": avg_words_per_page,
                    "complexity": complexity
                }
            else:
                return {"error": f"File {file_path} not found"}
    except Exception as e:
        logger.error(f"Error analyzing document complexity: {str(e)}")
        return {"error": str(e)}

async def route_to_appropriate_agent(tool_context:ToolContext, document_info: Dict[str, Any]) -> Dict:
    """
    Route a document to the appropriate processing agent based on its characteristics.

    Args:
        tool_context (ToolContext): The tool context containing state and artifacts.
        document_info (Dict[str,Any]): Dictionary containing document information including:
            - filename (str): Name of the document
            - complexity (str, optional): Document complexity ('low', 'medium', 'high')
            - Other metadata as needed

    Returns:
        Dict[str, Any]: A dictionary containing:
            - If successful: {
                'status': 'success',
                'message': str,
                'target_agent': str,  # Name of the target agent
                'processing_strategy': str  # Recommended processing strategy
              }
            - If error: {'status': 'error', 'message': str}"""
    try:
        complexity = document_info.get("complexity", "low")
        
        # Update the session state with routing decision
        doc_id = document_info.get("filename")
        if doc_id:
            if 'documents' not in tool_context.state:
                tool_context.state['documents'] = {}
            
            if doc_id not in tool_context.state['documents']:
                tool_context.state['documents'][doc_id] = {}
            
            context.state['documents'][doc_id]['complexity'] = complexity
            context.state['documents'][doc_id]['routed_to'] = "ingestion_agent"  # Default routing
            
            # Set up document processing workflow based on complexity
            if complexity == "high":
                # For complex documents, we might need more specialized processing
                context.state['documents'][doc_id]['processing_strategy'] = "detailed"
            else:
                context.state['documents'][doc_id]['processing_strategy'] = "standard"
        
        # For now, all documents go to the ingestion agent first
        return "Document routed to ingestion_agent for initial processing"
    except Exception as e:
        logger.error(f"Error routing document: {str(e)}")
        return f"Error routing document: {str(e)}"

# async def evaluate_agent_performance(tool_context:ToolContext, agent_id: str) -> Dict[str]:
#     """
#     Evaluate the performance of an agent on a specific task.

#     Args:
#         tool_context (ToolContext): The tool context containing state and artifacts.
#         agent_id (str): ID of the agent being evaluated.

#     Returns:
#         Dict[str]: A dictionary containing:
#             - If successful: {
#                 'status': 'success',
#                 'evaluation_id': str,
#                 'agent_id': str,
#                 'success': bool,
#                 'metrics': Dict[str]  # Performance metrics
#               }
#             - If error: {'status': 'error', 'message': str}"""
#     try:
#         # Simple evaluation logic - in a real system this would be more sophisticated
#         success = True
#         if isinstance(task_result, Dict) and "error" in task_result:
#             success = False
        
#         # Record the evaluation in the session state
#         if 'agent_evaluations' not in context.state:
#             context.state['agent_evaluations'] = {}
        
#         if agent_id not in context.state['agent_evaluations']:
#             context.state['agent_evaluations'][agent_id] = []
        
#         evaluation = {
#             "timestamp": datetime.now().isoformat(),
#             "success": success,
#             "result_summary": str(task_result)[:100] + "..." if len(str(task_result)) > 100 else str(task_result)
#         }
        
#         context.state['agent_evaluations'][agent_id].append(evaluation)
        
#         return {
#             "agent_id": agent_id,
#             "success": success,
#             "timestamp": evaluation["timestamp"]
#         }
#     except Exception as e:
#         logger.error(f"Error evaluating agent performance: {str(e)}")
#         return {"error": str(e)}

# async def process_user_feedback(tool_context:ToolContext, feedback: str, task_id: str) -> Dict[str]:
#     """
#     Process user feedback on agent performance or results.

#     Args:
#         tool_context (ToolContext): The tool context containing state and artifacts.
#         feedback (str): The feedback provided by the user.
#         task_id (str): ID of the task the feedback is related to.

#     Returns:
#         Dict[str]: A dictionary containing:
#             - If successful: {
#                 'status': 'success',
#                 'feedback_id': str,
#                 'task_id': str,
#                 'processed': bool,
#                 'action_taken': str  # Description of any actions taken
#               }
#             - If error: {'status': 'error', 'message': str}"""
#     try:
#         # Store the feedback in the session state
#         if 'user_feedback' not in tool_context.state:
#             tool_context.state['user_feedback'] = {}
        
#         tool_context.state['user_feedback'][task_id] = {
#             "feedback": feedback,
#             "timestamp": datetime.now().isoformat()
#         }
        
#         return {
#             "message": "Feedback recorded successfully",
#             "task_id": task_id,
#             "timestamp": context.state['user_feedback'][task_id]["timestamp"]
#         }
#     except Exception as e:
#         logger.error(f"Error processing user feedback: {str(e)}")
#         return {"error": str(e)}

# # Visualization Tools
#656
# async def create_comparison_visualization(tool_context:ToolContext, comparison_result: Dict) -> str:
#     """
#     Create and store a visualization of paper comparisons as an artifact.

#     This function takes comparison results between research papers and generates a structured
#     visualization artifact that can be used for analysis or display purposes. The visualization
#     data includes the comparison results along with metadata.

#     Args:
#         tool_context (ToolContext): The context object providing access to the current
#             execution environment, including artifact storage.
#         comparison_result (Dict): A dictionary containing the comparison data between papers.
#             Expected to include:
#                 - papers: List of paper identifiers being compared
#                 - comparison_metrics: Dictionary of metrics used for comparison
#                 - results: Nested structure with comparison details
#                 - Any additional comparison-specific data

#     Returns:
#         str: A status message indicating success or failure, including the name of the
#              generated artifact if successful.

#     Raises:
#         json.JSONEncodeError: If there's an error serializing the visualization data.
#         IOError: If there's an error saving the artifact.

#     Example:
#         > comparison_data = {
#         >     "papers": ["paper1", "paper2"],
#         >     "comparison_metrics": ["methodology", "results", "conclusions"],
#         >     "results": {
#         >         "methodology": {"similarity": 0.75, "details": "..."},
#         >         "results": {"similarity": 0.60, "details": "..."},
#         >         "conclusions": {"similarity": 0.85, "details": "..."}
#         >     }
#         > }
#         > await create_comparison_visualization(tool_context, comparison_data)
#         > 'Visualization created and saved as artifact: comparison_viz_20230604123045'

#     Notes:
#         - The function currently stores the visualization as a JSON artifact. In a production
#           environment, this could be extended to generate actual chart images or interactive
#           visualizations.
#         - The artifact name includes a timestamp to ensure uniqueness.
#     """
#     try:
#         # In a real implementation, this would generate a chart or graph
#         # For now, we'll just save the comparison data as an artifact
        
#         visualization_data = {
#             "type": "comparison_visualization",
#             "data": comparison_result,
#             "created_at": datetime.now().isoformat()
#         }
        
#         # Save as artifact
#         viz_artifact_name = f"comparison_viz_{datetime.now().strftime('%Y%m%d%H%M%S')}"
#         tool_context.save_artifact(viz_artifact_name, json.dumps(visualization_data).encode('utf-8'))
        
#         return f"Visualization created and saved as artifact: {viz_artifact_name}"
#     except Exception as e:
#         logger.error(f"Error creating visualization: {str(e)}")
#         return f"Error creating visualization: {str(e)}"
async def create_comparison_visualization(tool_context: ToolContext, comparison_result: List) -> str:
    """
    Create and store a visualization of paper comparisons as an artifact.

    This function takes a list of comparison results between research papers and generates a structured
    visualization artifact for analysis or display purposes. The visualization data includes the comparison
    results along with metadata.

    Args:
        tool_context (ToolContext): The context object providing access to the current
            execution environment, including artifact storage.
        comparison_result (List): A list of comparison data entries between papers. Each entry
            is expected to be a dictionary containing:
                - papers: List of paper identifiers being compared
                - comparison_metrics: Dictionary of metrics used for comparison
                - results: Nested structure with comparison details
                - Any additional comparison-specific data

    Returns:
        str: A status message indicating success or failure, including the name of the
             generated artifact if successful.

    Raises:
        json.JSONEncodeError: If there's an error serializing the visualization data.
        IOError: If there's an error saving the artifact.

    Example:
        >>> comparison_data = [
        >>>     {
        >>>         "papers": ["paper1", "paper2"],
        >>>         "comparison_metrics": ["methodology", "results", "conclusions"],
        >>>         "results": {
        >>>             "methodology": {"similarity": 0.75, "details": "..."},
        >>>             "results": {"similarity": 0.60, "details": "..."},
        >>>             "conclusions": {"similarity": 0.85, "details": "..."}
        >>>         }
        >>>     },
        >>>     {
        >>>         "papers": ["paper2", "paper3"],
        >>>         "comparison_metrics": ["methodology", "results"],
        >>>         "results": {
        >>>             "methodology": {"similarity": 0.70, "details": "..."},
        >>>             "results": {"similarity": 0.65, "details": "..."}
        >>>         }
        >>>     }
        >>> ]
        >>> await create_comparison_visualization(tool_context, comparison_data)
        'Visualization created and saved as artifact: comparison_viz_20250605162745'

    Notes:
        - The function stores the visualization as a JSON artifact. In a production environment,
          this could be extended to generate actual chart images or interactive visualizations.
        - The artifact name includes a timestamp to ensure uniqueness.
        - Using a List allows for multiple comparison entries, making it suitable for batch processing
          or comparisons across multiple paper pairs.
    """
    try:
        # Validate that comparison_result is a non-empty list
        if not isinstance(comparison_result, list) or not comparison_result:
            return "Error: comparison_result must be a non-empty list"

        # In a real implementation, this could generate a chart or graph
        # For now, we'll save the list of comparison data as an artifact
        visualization_data = {
            "type": "comparison_visualization",
            "data": comparison_result,
            "created_at": datetime.now().isoformat()
        }

        # Save as artifact
        viz_artifact_name = f"comparison_viz_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        viz=tool_context.save_artifact(filename=viz_artifact_name, artifact=types.Part(inline_data=types.Blob(mime_type="application/json",data=json.dumps(visualization_data).encode('utf-8'))))

        return f"Visualization created and saved as artifact: {viz_artifact_name}"
    except json.JSONEncodeError as e:
        logger.error(f"Error serializing visualization data: {str(e)}")
        return f"Error creating visualization: JSON serialization failed - {str(e)}"
    except IOError as e:
        logger.error(f"Error saving artifact: {str(e)}")
        return f"Error creating visualization: Failed to save artifact - {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error creating visualization: {str(e)}")
        return f"Error creating visualization: {str(e)}"
# ----------------------
# Agent Definitions
# ----------------------

# Create tool instances with proper documentation
extract_text_tool = FunctionTool(extract_text_from_pdf)

extract_metadata_tool = FunctionTool(extract_metadata_from_pdf)

process_directory_tool = FunctionTool(
    process_pdf_directory
)

identify_topics_tool = FunctionTool(
    identify_common_topics
)

compare_methodologies_tool = FunctionTool(
    compare_methodologies
)

compare_findings_tool = FunctionTool(
    compare_findings
)

generate_analysis_tool = FunctionTool(generate_cross_paper_analysis
)

analyze_complexity_tool = FunctionTool(analyze_document_complexity
)

route_document_tool = FunctionTool(route_to_appropriate_agent
)

# evaluate_performance_tool = FunctionTool(evaluate_agent_performance
# )

# process_feedback_tool = FunctionTool(process_user_feedback
# )

create_visualization_tool = FunctionTool(create_comparison_visualization
)

# Define agent instructions
INGESTION_AGENT_INSTRUCTION = """
You are the Ingestion Agent for the ScholarVerse research paper comparison system. Your job is to:

1. Process PDF files by extracting their text and metadata
2. Store the extracted information as artifacts
3. Report on the success or failure of the ingestion process

When given a path to a PDF file or a directory containing PDF files:
- If given a single PDF, extract its text and metadata
- If given a directory, process all PDF files in that directory
- Store the results as artifacts for other agents to use

Always confirm when a document has been processed and provide details about any errors encountered.
"""

CROSS_PAPER_AGENT_INSTRUCTION = """
You are the Cross-Paper Analysis Agent for ScholarVerse. Your job is to compare multiple research papers and identify:

1. Common topics, themes, and research areas
You have to utilize identify topics tool to identify common topics, themes, and research areas

Your output should help researchers understand how multiple papers relate to each other in a specific research domain.
"""
#2. Similarities and differences in methodologies
# 3. Complementary or contradictory findings
# 4. Potential avenues for future research based on the comparative analysis
#
#
#Work with the text extracted by the Ingestion Agent, stored as artifacts. Your analysis should be:
#- Structured and organized by categories (topics, methods, findings)
#- Objective, highlighting factual comparisons
#- Insightful, identifying patterns that might not be immediately obvious
#- Comprehensive, covering all papers in the comparison
ROUTER_AGENT_INSTRUCTION = """
You are the main Router Agent for ScholarVerse, responsible for coordinating the flow between specialized sub-agents that handle different aspects of scientific papers.

State Management:
- Track document processing status in state['document_status']
- Maintain analysis results in state['analysis_results']
- Store user preferences in state['user_preferences']
- Track interaction history in state['interaction_history']

Available Sub-Agents:
1. Ingestion Agent: Processes and extracts information from uploaded documents
2. Cross-Paper Analysis Agent: Performs comparative analysis across multiple papers



Workflow:
1. When a new document is uploaded, route it to the Ingestion Agent
2. After ingestion, coordinate analysis with appropriate sub-agents
3. Use the analysis results and return them to user

Always maintain context and state between interactions to provide coherent responses.
"""
#3. Insight Agent: Generates AI-powered insights and summaries
#4. Visualization Agent: Creates interactive visualizations of the knowledge graph
#
# 4. Generate insights and visualizations based on user queries
INSIGHT_AGENT_INSTRUCTION = """
You are the Insight Agent for ScholarVerse. Your job is to generate insightful analyses of research papers and their comparisons. You should:

1. Identify key takeaways from individual papers
2. Highlight important connections between papers
3. Suggest potential implications or applications of the research
4. Identify gaps or opportunities for future research

Work with the documents processed by the Ingestion Agent and the comparisons from the Cross-Paper Analysis Agent. Your insights should be:
- Clear and concise
- Evidence-based, citing specific content from the papers
- Valuable for researchers in the field
- Organized in a way that highlights the most important points first

Focus on providing actionable insights that would help researchers understand the significance of the papers in their field.
"""

VISUALIZATION_AGENT_INSTRUCTION = """
You are the Visualization Agent for ScholarVerse. Your job is to create visualizations that help researchers understand the relationships between papers and their content. You should:

1. Generate visualizations of paper comparisons
2. Create visual representations of research topics and their connections
3. Visualize methodological approaches across papers
4. Illustrate findings and their relationships

Work with the data processed by other agents to create meaningful visualizations. Your visualizations should be:
- Clear and easy to interpret
- Focused on highlighting important patterns or relationships
- Appropriate for the type of data being visualized
- Helpful for researchers trying to understand the literature

Describe the visualizations you would create and store them as artifacts when possible.
"""
#model="gemini-2.0-flash",
# Create the agents
ingestion_agent = Agent(
    name="ingestion_agent",
    model=model,
    description="Agent responsible for ingesting and extracting structured information from research papers",
    instruction=INGESTION_AGENT_INSTRUCTION,
    tools=[
        extract_text_tool,
        extract_metadata_tool,
        process_directory_tool
    ]
)

cross_paper_agent = Agent(
    name="cross_paper_agent",
    model=model,
    description="Agent responsible for comparing multiple research papers and identifying relationships",
    instruction=CROSS_PAPER_AGENT_INSTRUCTION,
    tools=[
        identify_topics_tool,

    ]
)
#        compare_methodologies_tool,
#   compare_findings_tool,
#    generate_analysis_tool
insight_agent = Agent(
    name="insight_agent",
    model=model,
    description="Agent responsible for generating insights and summaries from research papers",
    instruction=INSIGHT_AGENT_INSTRUCTION,
    tools=[]  # This agent primarily uses the LLM's reasoning capabilities
)

visualization_agent = Agent(
    name="visualization_agent",
    model=model,
    description="Agent responsible for creating visualizations of paper comparisons and relationships",
    instruction=VISUALIZATION_AGENT_INSTRUCTION,
    tools=[
        create_visualization_tool
    ]
)

# Create a sequential agent for the document processing pipeline  visualization_agent is not included, , insight_agent
document_processing_pipeline = SequentialAgent(
    name="document_processing_pipeline",
    sub_agents=[ingestion_agent, cross_paper_agent],
    description="Sequential pipeline for processing documents from ingestion to visualization"
)

# Create the main router agent
root_agent = Agent(
    name="router_agent",
    model=model,
    description="Main orchestrator agent for ScholarVerse that routes tasks to specialized agents",
    instruction=ROUTER_AGENT_INSTRUCTION,
    sub_agents=[document_processing_pipeline],
    tools=[
        analyze_complexity_tool,
        route_document_tool,
        
    ]
)

# ----------------------
# Runner and Session Setup
# ----------------------

def setup_agent_environment(app_name="paper_comparison", user_id="23233", session_id="12323231"):
    """Set up the agent environment with InMemoryArtifactService."""
    # Set up the session service
    session_service = InMemorySessionService()
    session = session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id
    )
    
    # Set up the runner
    runner = Runner(
        app_name=app_name,
        agent=root_agent,
        session_service=session_service,
        artifact_service=InMemoryArtifactService()  # Using InMemoryArtifactService for PDF storage
    )
    return runner, session

def process_folder(folder_path, runner, session):
    """Process all PDF files in a folder."""
    content = types.Content(
        role="user",
        parts=[types.Part(text=f"Process all the research papers in this folder: {folder_path}")]
    )
    events = runner.run(
        user_id=session.user_id,
        session_id=session.id,
        new_message=content
    )
    responses = []
    for event in events:
        if event.is_final_response():
            response = event.content.parts[0].text
            responses.append(response)
            print(response)
    
    return responses

def compare_papers(comparison_type, runner, session):
    """Perform a specific type of comparison on the processed papers."""
    content = types.Content(
        role="user",
        parts=[types.Part(text=f"Perform a {comparison_type} comparison on the papers I've uploaded.")]
    )
    
    events = runner.run(
        user_id=session.user_id,
        session_id=session.id,
        new_message=content
    )
    responses=[]
    for event in events:
        if event.is_final_response():
            response = event.content.parts[0].text
            responses.append(response)
    
    return responses
# Set up the environment
async def main():
    runner, session = setup_agent_environment(
    app_name="scholar_verse", 
    user_id="researcher_1", 
    session_id="1234"
)
# Process a folder
    # Process a folder of research papers
    print('dadsaad')
    process_folder("D:/downloads/research_papers", runner, session)
    print('aadcsad')
    # # Perform a specific type of comparison
    # compare_papers("methodology", runner, session)
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
    import asyncio
    asyncio.run(main())