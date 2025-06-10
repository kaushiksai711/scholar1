from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import hashlib
import json

@dataclass
class ResearchDocument:
    """Class to represent a research document with metadata and content."""
    raw_text: str
    doc_id: Optional[str] = None
    title: str = ""
    authors: List[str] = field(default_factory=list)
    publication_date: Optional[datetime] = None
    metadata: Dict = field(default_factory=dict)
    
    # Extracted sections
    abstract: str = ""
    keywords: List[str] = field(default_factory=list)
    introduction: str = ""
    methodology: str = ""
    references: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        """Process the document after initialization."""
        if not self.doc_id:
            self.doc_id = self._generate_id()
        
        # Extract metadata if not provided
        if not self.abstract or not self.keywords:
            self._extract_metadata()
    
    def _generate_id(self) -> str:
        """Generate a unique ID for the document."""
        content = f"{self.title}{self.raw_text[:1000]}".encode('utf-8')
        return hashlib.md5(content).hexdigest()
    
    def _extract_metadata(self):
        """Extract metadata from raw text."""
        from research_utils import extract_research_metadata
        
        metadata = extract_research_metadata(self.raw_text)
        self.abstract = metadata.get('abstract', '')
        self.keywords = [k.strip() for k in metadata.get('keywords', '').split(';') if k.strip()]
        self.introduction = metadata.get('introduction', '')
        self.methodology = metadata.get('methodology', '')
    
    def to_dict(self) -> Dict:
        """Convert document to dictionary."""
        return {
            'doc_id': self.doc_id,
            'title': self.title,
            'authors': self.authors,
            'publication_date': self.publication_date.isoformat() if self.publication_date else None,
            'abstract': self.abstract,
            'keywords': self.keywords,
            'introduction': self.introduction,
            'methodology': self.methodology,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ResearchDocument':
        """Create document from dictionary."""
        doc = cls(
            raw_text=data.get('raw_text', ''),
            doc_id=data.get('doc_id'),
            title=data.get('title', ''),
            authors=data.get('authors', []),
            publication_date=datetime.fromisoformat(data['publication_date']) if data.get('publication_date') else None,
            metadata=data.get('metadata', {})
        )
        
        # Set extracted fields if available
        if 'abstract' in data:
            doc.abstract = data['abstract']
        if 'keywords' in data:
            doc.keywords = data['keywords']
        if 'introduction' in data:
            doc.introduction = data['introduction']
        if 'methodology' in data:
            doc.methodology = data['methodology']
            
        return doc
    
    def __str__(self) -> str:
        return f"ResearchDocument(id={self.doc_id}, title={self.title})"