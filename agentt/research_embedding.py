import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import numpy.typing as npt
import logging
from dataclasses import dataclass
from research_document import ResearchDocument

logger = logging.getLogger(__name__)

@dataclass
class DocumentEmbedding:
    """Class to handle document embeddings and similarity calculations."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize with a pre-trained sentence transformer model.
        
        Args:
            model_name: Name of the pre-trained model. Recommended options:
                       - 'all-MiniLM-L6-v2': General purpose, fast
                       - 'allenai/specter': Research paper specific
                       - 'sentence-transformers/all-mpnet-base-v2': Higher quality, slower
        """
        self.model = SentenceTransformer(model_name)
        self.embedding_cache: Dict[str, npt.NDArray] = {}
    
    def embed_document(self, doc: ResearchDocument) -> npt.NDArray:
        """Generate embedding for a research document."""
        cache_key = doc.doc_id
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # Create a structured text representation for embedding
        text_parts = [
            doc.title,
            doc.abstract,
            ' '.join(doc.keywords),
            doc.introduction[:1000],  # First 1000 chars of introduction
            doc.methodology[:1000]    # First 1000 chars of methodology
        ]
        text = ' '.join(filter(None, text_parts))
        
        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)
        self.embedding_cache[cache_key] = embedding
        return embedding
    
    def batch_embed_documents(self, docs: List[ResearchDocument]) -> npt.NDArray:
        """Generate embeddings for multiple documents."""
        return np.array([self.embed_document(doc) for doc in docs])
    
    def calculate_pairwise_similarity(
        self, 
        doc1: ResearchDocument, 
        doc2: ResearchDocument
    ) -> Dict[str, float]:
        """
        Calculate multiple similarity metrics between two research documents.
        
        Returns:
            Dictionary containing various similarity scores
        """
        # Get document embeddings
        emb1 = self.embed_document(doc1)
        emb2 = self.embed_document(doc2)
        
        # Calculate cosine similarity between document embeddings
        semantic_sim = cosine_similarity([emb1], [emb2])[0][0]
        
        # Calculate keyword overlap (Jaccard similarity)
        keywords1 = set(kw.lower() for kw in doc1.keywords)
        keywords2 = set(kw.lower() for kw in doc2.keywords)
        keyword_overlap = (
            len(keywords1 & keywords2) / max(1, len(keywords1 | keywords2))
            if keywords1 or keywords2 else 0.0
        )
        
        # Calculate title word overlap
        title_words1 = set(doc1.title.lower().split())
        title_words2 = set(doc2.title.lower().split())
        title_overlap = (
            len(title_words1 & title_words2) / max(1, len(title_words1 | title_words2))
            if title_words1 and title_words2 else 0.0
        )
        
        # Combine scores (weighted average)
        combined_score = (
            0.6 * semantic_sim + 
            0.25 * keyword_overlap + 
            0.15 * title_overlap
        )
        
        return {
            'semantic_similarity': float(semantic_sim),
            'keyword_overlap': float(keyword_overlap),
            'title_similarity': float(title_overlap),
            'combined_score': float(combined_score)
        }
    
    def cluster_documents(
        self,
        docs: List[ResearchDocument],
        min_cluster_size: int = 2,
        threshold: float = 0.7
    ) -> List[List[ResearchDocument]]:
        """
        Cluster similar research documents using HDBSCAN.
        
        Args:
            docs: List of ResearchDocument objects
            min_cluster_size: Minimum number of documents in a cluster
            threshold: Similarity threshold for clustering (0-1)
            
        Returns:
            List of document clusters, where each cluster is a list of similar documents
        """
        if len(docs) < 2:
            return [docs]
            
        # Get document embeddings
        embeddings = self.batch_embed_documents(docs)
        
        # Use DBSCAN for density-based clustering
        clustering = DBSCAN(
            eps=1.0 - threshold,  # Convert similarity to distance
            min_samples=min_cluster_size,
            metric='cosine'
        ).fit(embeddings)
        
        # Group documents by cluster
        clusters: Dict[int, List[ResearchDocument]] = {}
        for idx, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(docs[idx])
        
        # Convert to list of clusters, excluding noise points (label = -1)
        result = [cluster for label, cluster in clusters.items() if label != -1]
        
        # Add unclustered documents as single-document clusters
        unclustered = clusters.get(-1, [])
        result.extend([[doc] for doc in unclustered])
        
        return result