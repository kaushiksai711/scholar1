from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from research_document import ResearchDocument
from research_embedding import DocumentEmbedding
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import logging
from collections import Counter
import re

logger = logging.getLogger(__name__)

@dataclass
class ResearchTopic:
    """Represents a research topic with associated documents and keywords."""
    name: str
    keywords: List[Tuple[str, float]]  # (keyword, score) pairs
    documents: List[ResearchDocument]
    parent_topic: Optional['ResearchTopic'] = None
    subtopics: List['ResearchTopic'] = None
    
    def __post_init__(self):
        self.subtopics = self.subtopics or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert topic to dictionary for serialization."""
        return {
            'name': self.name,
            'keywords': self.keywords,
            'document_ids': [doc.doc_id for doc in self.documents],
            'subtopics': [t.to_dict() for t in self.subtopics]
        }
    
    @property
    def document_count(self) -> int:
        """Get total number of documents in this topic and subtopics."""
        return len(self.documents) + sum(len(t.documents) for t in self.subtopics)

class ResearchTopicModeler:
    """Handles topic modeling for research documents."""
    
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.embedder = DocumentEmbedding(embedding_model)
        self.vectorizer = TfidfVectorizer(
            max_df=0.95, 
            min_df=2,
            stop_words='english',
            ngram_range=(1, 2)  # Include bigrams
        )
    
    def extract_topics_lda(
        self, 
        documents: List[ResearchDocument],
        n_topics: int = 5,
        n_keywords: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Extract topics using LDA.
        
        Args:
            documents: List of research documents
            n_topics: Number of topics to extract
            n_keywords: Number of keywords per topic
            
        Returns:
            List of topic dictionaries with keywords and document indices
        """
        # Combine text from abstract, keywords, and introduction
        texts = [
            f"{doc.abstract} {' '.join(doc.keywords)} {doc.introduction[:1000]}"
            for doc in documents
        ]
        
        # Create TF-IDF features
        tfidf = self.vectorizer.fit_transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Apply LDA
        lda = LatentDirichletAllocation(
            n_components=min(n_topics, len(documents) - 1),
            random_state=42,
            learning_method='online'
        )
        lda.fit(tfidf)
        
        # Extract topics
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            # Get top keywords for this topic
            top_keywords_idx = topic.argsort()[:-n_keywords - 1:-1]
            keywords = [(feature_names[i], float(topic[i])) 
                       for i in top_keywords_idx]
            
            # Get documents most relevant to this topic
            doc_topic_probs = lda.transform(tfidf)[:, topic_idx]
            top_doc_indices = doc_topic_probs.argsort()[-5:][::-1]  # Top 5 documents
            
            topics.append({
                'topic_id': topic_idx,
                'keywords': keywords,
                'document_indices': top_doc_indices.tolist(),
                'documents': [documents[i] for i in top_doc_indices]
            })
        
        return topics
    
    def hierarchical_topic_modeling(
        self,
        documents: List[ResearchDocument],
        max_levels: int = 3,
        min_docs_per_topic: int = 3
    ) -> ResearchTopic:
        """
        Perform hierarchical topic modeling on research documents.
        
        Args:
            documents: List of research documents
            max_levels: Maximum depth of the topic hierarchy
            min_docs_per_topic: Minimum documents required to form a topic
            
        Returns:
            Root ResearchTopic with hierarchical subtopics
        """
        root_topic = ResearchTopic(
            name="Root",
            keywords=[],
            documents=[],
            parent_topic=None
        )
        
        self._build_topic_hierarchy(
            documents=documents,
            parent_topic=root_topic,
            current_level=0,
            max_levels=max_levels,
            min_docs=min_docs_per_topic
        )
        
        return root_topic
    
    def _build_topic_hierarchy(
        self,
        documents: List[ResearchDocument],
        parent_topic: ResearchTopic,
        current_level: int,
        max_levels: int,
        min_docs: int
    ) -> None:
        """Recursively build topic hierarchy."""
        if len(documents) < min_docs or current_level >= max_levels:
            parent_topic.documents.extend(documents)
            return
            
        # Cluster documents
        clusters = self.embedder.cluster_documents(
            documents,
            min_cluster_size=min_docs
        )
        
        # If only one cluster or all in one cluster, try to split further
        if len(clusters) <= 1:
            # Try LDA-based topic modeling
            topics = self.extract_topics_lda(
                documents,
                n_topics=min(3, max(2, len(documents) // 2))
            )
            
            if len(topics) > 1:
                # Create subtopics based on LDA
                for topic in topics:
                    topic_docs = topic['documents']
                    if len(topic_docs) >= min_docs:
                        topic_name = ' '.join([kw[0] for kw in topic['keywords'][:3]])
                        subtopic = ResearchTopic(
                            name=topic_name,
                            keywords=topic['keywords'],
                            documents=[],
                            parent_topic=parent_topic
                        )
                        parent_topic.subtopics.append(subtopic)
                        
                        # Recursively process this subtopic
                        self._build_topic_hierarchy(
                            documents=topic_docs,
                            parent_topic=subtopic,
                            current_level=current_level + 1,
                            max_levels=max_levels,
                            min_docs=min_docs
                        )
                    else:
                        parent_topic.documents.extend(topic_docs)
            else:
                parent_topic.documents.extend(documents)
        else:
            # Process each cluster
            for i, cluster in enumerate(clusters):
                if len(cluster) >= min_docs:
                    # Extract common keywords for cluster name
                    cluster_keywords = self._extract_common_keywords(cluster)
                    topic_name = f"Topic {i+1}: {' '.join(kw[0] for kw in cluster_keywords[:3])}"
                    
                    subtopic = ResearchTopic(
                        name=topic_name,
                        keywords=cluster_keywords,
                        documents=[],
                        parent_topic=parent_topic
                    )
                    parent_topic.subtopics.append(subtopic)
                    
                    # Recursively process this subtopic
                    self._build_topic_hierarchy(
                        documents=cluster,
                        parent_topic=subtopic,
                        current_level=current_level + 1,
                        max_levels=max_levels,
                        min_docs=min_docs
                    )
                else:
                    parent_topic.documents.extend(cluster)
    
    def _extract_common_keywords(
        self,
        documents: List[ResearchDocument],
        top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """Extract common keywords from a set of documents."""
        # Collect all keywords
        all_keywords = []
        for doc in documents:
            all_keywords.extend(kw.lower() for kw in doc.keywords)
        
        # Count keyword frequencies
        keyword_counts = Counter(all_keywords)
        total_keywords = sum(keyword_counts.values())
        
        # Calculate TF-IDF like scores
        scores = []
        for kw, count in keyword_counts.most_common(top_n * 2):  # Get more to filter
            # Simple TF-IDF like score (TF * log(N/df))
            tf = count / total_keywords
            df = sum(1 for d in documents if any(kw == k.lower() for k in d.keywords))
            idf = np.log(len(documents) / (df + 1))  # Add 1 to avoid division by zero
            score = tf * idf
            scores.append((kw, score))
        
        # Sort by score and return top N
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]