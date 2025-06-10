from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from research_document import ResearchDocument
from research_embedding import DocumentEmbedding
import numpy as np
from collections import defaultdict
import logging
import re

logger = logging.getLogger(__name__)

@dataclass
class ResearchGap:
    """Represents a potential research gap or opportunity."""
    description: str
    supporting_evidence: List[str]
    related_documents: List[ResearchDocument]
    confidence: float
    gap_type: str  # e.g., 'methodological', 'theoretical', 'empirical', 'application'
    keywords: List[str]
    
    def to_dict(self) -> Dict:
        """Convert gap to dictionary for serialization."""
        return {
            'description': self.description,
            'supporting_evidence': self.supporting_evidence,
            'document_ids': [doc.doc_id for doc in self.related_documents],
            'confidence': self.confidence,
            'gap_type': self.gap_type,
            'keywords': self.keywords
        }

class ResearchGapAnalyzer:
    """Identifies research gaps and opportunities in a collection of papers."""
    
    def __init__(self, embedder: DocumentEmbedding):
        self.embedder = embedder
        self.method_phrases = [
            "future work should", "further research is needed", "limitation of this study",
            "open question", "remains unclear", "not well understood", "lacks investigation",
            "requires further investigation", "opportunity for future research",
            "suggest future research", "directions for future research", "research gap",
            "further studies are needed", "remains to be explored", "warrants further investigation"
        ]
        
    def identify_gaps(
        self, 
        documents: List[ResearchDocument],
        min_support: int = 2
    ) -> List[ResearchGap]:
        """
        Identify research gaps from a collection of papers.
        
        Args:
            documents: List of research documents
            min_support: Minimum number of documents that should mention a gap for it to be considered
            
        Returns:
            List of identified research gaps
        """
        # Extract explicit gap mentions
        explicit_gaps = self._find_explicit_gaps(documents)
        
        # Identify implicit gaps through content analysis
        implicit_gaps = self._find_implicit_gaps(documents)
        
        # Combine and deduplicate gaps
        all_gaps = self._combine_gaps(explicit_gaps + implicit_gaps, min_support)
        
        # Score and sort gaps
        all_gaps = self._score_gaps(all_gaps, documents)
        
        return all_gaps
    
    def _find_explicit_gaps(self, documents: List[ResearchDocument]) -> List[ResearchGap]:
        """Find explicitly mentioned gaps in conclusions and future work sections."""
        gaps = []
        
        for doc in documents:
            # Check conclusion and future work sections
            text = f"{doc.introduction} {doc.methodology}"
            if not text.strip():
                continue
                
            # Find gap-indicating phrases
            for phrase in self.method_phrases:
                if phrase.lower() in text.lower():
                    # Extract context around the gap mention
                    context = self._extract_context(text, phrase, window=200)
                    gap = ResearchGap(
                        description=f"Explicit gap mentioned: {phrase}",
                        supporting_evidence=[context],
                        related_documents=[doc],
                        confidence=0.8,  # High confidence for explicit mentions
                        gap_type="explicit",
                        keywords=self._extract_keywords(context)
                    )
                    gaps.append(gap)
        
        return gaps
    
    def _find_implicit_gaps(
        self, 
        documents: List[ResearchDocument],
        similarity_threshold: float = 0.7
    ) -> List[ResearchGap]:
        """Identify implicit gaps through content analysis."""
        if len(documents) < 2:
            return []
            
        # Cluster documents to find related papers
        clusters = self.embedder.cluster_documents(
            documents,
            min_cluster_size=2,
            threshold=similarity_threshold
        )
        
        gaps = []
        
        # Analyze each cluster for potential gaps
        for cluster in clusters:
            if len(cluster) < 2:
                continue
                
            # Find common and unique aspects in the cluster
            common_keywords = self._get_common_keywords(cluster)
            unique_keywords_by_doc = self._get_unique_keywords_by_doc(cluster)
            
            # Identify potential gaps based on unique aspects
            for doc, unique_keywords in unique_keywords_by_doc.items():
                if not unique_keywords:
                    continue
                    
                # Create a gap for unique aspects that could be opportunities
                gap_desc = (
                    f"Potential gap: {', '.join(unique_keywords[:3])} "
                    f"in {doc.title} differs from other works in this cluster"
                )
                
                gap = ResearchGap(
                    description=gap_desc,
                    supporting_evidence=[
                        f"Document focuses on: {', '.join(unique_keywords[:5])}",
                        f"While other works in this cluster focus on: {', '.join(common_keywords[:5])}"
                    ],
                    related_documents=cluster,
                    confidence=0.6,  # Moderate confidence for implicit gaps
                    gap_type="implicit",
                    keywords=unique_keywords[:5]
                )
                gaps.append(gap)
        
        return gaps
    
    def _extract_context(self, text: str, phrase: str, window: int = 200) -> str:
        """Extract context around a phrase in text."""
        text_lower = text.lower()
        phrase_lower = phrase.lower()
        start = text_lower.find(phrase_lower)
        
        if start == -1:
            return ""
            
        # Get the start and end indices for the context window
        start = max(0, start - window)
        end = min(len(text), start + len(phrase) + 2 * window)
        
        # Add ellipsis if we didn't get the full context
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(text) else ""
        
        return f"{prefix}{text[start:end].strip()}{suffix}"
    
    def _extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """Extract important keywords from text."""
        # Simple implementation - can be enhanced with more sophisticated NLP
        words = re.findall(r'\b\w+\b', text.lower())
        word_counts = {}
        for word in words:
            if len(word) > 3:  # Ignore very short words
                word_counts[word] = word_counts.get(word, 0) + 1
        return [w for w, _ in sorted(word_counts.items(), key=lambda x: -x[1])[:top_n]]
    
    def _get_common_keywords(
        self, 
        documents: List[ResearchDocument]
    ) -> List[str]:
        """Get keywords common to all documents in a cluster."""
        if not documents:
            return []
            
        # Get intersection of all keywords
        common_keywords = set(doc.keywords[0].lower() for doc in documents if doc.keywords)
        for doc in documents[1:]:
            if not doc.keywords:
                continue
            doc_keywords = set(kw.lower() for kw in doc.keywords)
            common_keywords &= doc_keywords
            if not common_keywords:
                break
                
        return list(common_keywords)
    
    def _get_unique_keywords_by_doc(
        self, 
        documents: List[ResearchDocument]
    ) -> Dict[ResearchDocument, List[str]]:
        """Get keywords unique to each document in a cluster."""
        if len(documents) < 2:
            return {}
            
        # Count keyword frequencies across all documents
        keyword_counts = defaultdict(int)
        for doc in documents:
            for kw in doc.keywords:
                keyword_counts[kw.lower()] += 1
                
        # Find keywords unique to each document
        unique_keywords = {}
        for doc in documents:
            doc_keywords = set(kw.lower() for kw in doc.keywords)
            unique = [
                kw for kw in doc_keywords 
                if keyword_counts.get(kw, 0) == 1 and len(kw) > 3
            ]
            unique_keywords[doc] = unique[:5]  # Limit to top 5 unique keywords
            
        return unique_keywords
    
    def _combine_gaps(
        self, 
        gaps: List[ResearchGap], 
        min_support: int
    ) -> List[ResearchGap]:
        """Combine similar gaps and filter by minimum support."""
        if not gaps:
            return []
            
        # Group gaps by their type and keywords
        gap_groups = defaultdict(list)
        for gap in gaps:
            key = (gap.gap_type, tuple(sorted(gap.keywords))[:3])
            gap_groups[key].append(gap)
            
        # Combine gaps in the same group
        combined_gaps = []
        for (gap_type, _), group in gap_groups.items():
            if len(group) < min_support and gap_type != "explicit":
                continue
                
            # Combine supporting evidence and documents
            all_evidence = []
            all_docs = set()
            for gap in group:
                all_evidence.extend(gap.supporting_evidence)
                all_docs.update(gap.related_documents)
                
            # Create combined gap
            combined_gap = ResearchGap(
                description=group[0].description,
                supporting_evidence=all_evidence[:5],  # Limit evidence
                related_documents=list(all_docs),
                confidence=sum(g.confidence for g in group) / len(group),
                gap_type=gap_type,
                keywords=group[0].keywords
            )
            combined_gaps.append(combined_gap)
            
        return combined_gaps
    
    def _score_gaps(
        self, 
        gaps: List[ResearchGap], 
        all_documents: List[ResearchDocument]
    ) -> List[ResearchGap]:
        """Score and sort gaps by importance."""
        if not gaps:
            return []
            
        # Simple scoring - can be enhanced
        for gap in gaps:
            # Increase score for explicit gaps
            if gap.gap_type == "explicit":
                gap.confidence = min(1.0, gap.confidence * 1.2)
                
            # Increase score based on number of supporting documents
            doc_support = len(gap.related_documents) / len(all_documents)
            gap.confidence = min(1.0, gap.confidence * (1.0 + doc_support * 0.5))
            
        # Sort by confidence (descending)
        return sorted(gaps, key=lambda x: -x.confidence)