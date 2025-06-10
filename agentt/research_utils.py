import re
from typing import Dict, List, Tuple
import spacy

# Load spaCy model for better NLP processing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_research_metadata(text: str) -> Dict[str, str]:
    """Extract structured metadata from research paper text."""
    sections = {
        'abstract': '',
        'keywords': '',
        'introduction': '',
        'methodology': '',
        'conclusion': ''
    }
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Extract abstract
    abstract_match = re.search(
        r'(?i)abstract[:\s]*(.*?)(?=\n\s*(?:keywords|introduction|1\.|background|$))',
        text,
        re.DOTALL
    )
    if abstract_match:
        sections['abstract'] = abstract_match.group(1).strip()
    
    # Extract keywords
    keywords_match = re.search(
        r'(?i)keywords?[:\s]*(.*?)(?=\n\s*(?:1\.|introduction|$))',
        text,
        re.DOTALL
    )
    if keywords_match:
        sections['keywords'] = keywords_match.group(1).strip()
    
    # Extract introduction
    intro_match = re.search(
        r'(?i)(?:1\.|introduction)[:\s]*(.*?)(?=\n\s*(?:2\.|related work|methodology|$))',
        text,
        re.DOTALL
    )
    if intro_match:
        sections['introduction'] = intro_match.group(1).strip()
    
    # Extract methodology
    method_match = re.search(
        r'(?i)(?:2\.|methodology|methods)[:\s]*(.*?)(?=\n\s*(?:3\.|results|findings|$))',
        text,
        re.DOTALL
    )
    if method_match:
        sections['methodology'] = method_match.group(1).strip()
    
    return sections

def preprocess_research_text(text: str, preserve_case: bool = False) -> str:
    """Preprocess research paper text while preserving technical terms."""
    if not text:
        return ""
    
    # Remove citations
    text = re.sub(r'\[[0-9,\s-]+\]', '', text)  # [1], [1,2], [1-3]
    text = re.sub(r'\([^)]*?\d{4}[^)]*?\)', '', text)  # (Author et al., 2020)
    
    # Handle special characters and formatting
    text = re.sub(r'-\n', '', text)  # Handle hyphenated line breaks
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Tokenize and process with spaCy
    doc = nlp(text)
    tokens = []
    
    for token in doc:
        if token.is_space:
            continue
        if token.is_punct:
            continue
        if token.is_stop:
            continue
            
        # Lemmatize
        lemma = token.lemma_.lower() if not preserve_case else token.lemma_
        tokens.append(lemma)
    
    return ' '.join(tokens)

def extract_key_phrases(text: str, top_n: int = 10) -> List[Tuple[str, float]]:
    """Extract key phrases from text using noun chunks and their frequencies."""
    doc = nlp(text)
    noun_chunks = list(doc.noun_chunks)
    
    # Count frequencies of noun chunks
    chunk_freq = {}
    for chunk in noun_chunks:
        chunk_text = chunk.text.lower().strip()
        if len(chunk_text.split()) > 1:  # Only consider multi-word phrases
            chunk_freq[chunk_text] = chunk_freq.get(chunk_text, 0) + 1
    
    # Sort by frequency and return top N
    return sorted(chunk_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]