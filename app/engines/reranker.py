"""
Cross-Encoder Re-ranker - Precision re-ranking of search results
Uses cross-encoder models to score query-document pairs for high-accuracy ranking.
Inspired by: Cohere Rerank, ColBERT, MS MARCO
"""
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import logging
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

from typing import List, Dict, Tuple, Optional
import numpy as np

# Conditional import
try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False
    print("Warning: sentence-transformers CrossEncoder not available")


class Reranker:
    """
    Cross-Encoder based Re-ranker
    
    Unlike bi-encoders (used in retrieval), cross-encoders see query and document
    together, enabling much more accurate relevance scoring at the cost of speed.
    
    Use case: After initial retrieval (fast but approximate), rerank top-k results
    for high precision.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the reranker.
        
        Args:
            model_name: HuggingFace model name for cross-encoder.
                        Default is a fast, accurate model trained on MS MARCO.
        """
        self.model = None
        self.initialized = False
        self.model_name = model_name
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Load the cross-encoder model"""
        if not HAS_CROSS_ENCODER:
            print("[Reranker] Disabled - CrossEncoder not available")
            return
        
        try:
            self.model = CrossEncoder(self.model_name)
            self.initialized = True
            print(f"[Reranker] Initialized with model: {self.model_name}")
        except Exception as e:
            print(f"[Reranker] Failed to initialize: {e}")
            # Fallback: use a lighter model or disable
            try:
                self.model = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")
                self.initialized = True
                print("[Reranker] Fallback to TinyBERT model")
            except:
                self.initialized = False
    
    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[int, str, float]]:
        """
        Re-rank documents based on relevance to query.
        
        Args:
            query: The search query
            documents: List of document texts to re-rank
            top_k: Number of top results to return
            
        Returns:
            List of (original_index, document, score) tuples, sorted by relevance
        """
        if not self.initialized or not documents:
            # Fallback: return original order with dummy scores
            return [(i, doc, 1.0 - i*0.1) for i, doc in enumerate(documents[:top_k])]
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        try:
            # Score all pairs
            scores = self.model.predict(pairs)
            
            # Create indexed results
            indexed_results = [
                (i, documents[i], float(scores[i])) 
                for i in range(len(documents))
            ]
            
            # Sort by score (descending)
            indexed_results.sort(key=lambda x: x[2], reverse=True)
            
            return indexed_results[:top_k]
            
        except Exception as e:
            print(f"[Reranker] Error during reranking: {e}")
            return [(i, doc, 1.0 - i*0.1) for i, doc in enumerate(documents[:top_k])]
    
    def rerank_with_metadata(
        self, 
        query: str, 
        results: List[Dict], 
        text_key: str = "text",
        top_k: int = 5
    ) -> List[Dict]:
        """
        Re-rank results while preserving metadata.
        
        Args:
            query: The search query
            results: List of result dicts, each containing text and metadata
            text_key: Key in result dict containing the text to score
            top_k: Number of top results to return
            
        Returns:
            Reranked list of result dicts with added 'rerank_score' field
        """
        if not results:
            return []
        
        # Extract texts
        texts = [r.get(text_key, str(r)) for r in results]
        
        # Get reranked indices and scores
        reranked = self.rerank(query, texts, top_k=min(top_k, len(results)))
        
        # Build output with original metadata
        output = []
        for orig_idx, _, score in reranked:
            result = results[orig_idx].copy()
            result["rerank_score"] = score
            result["original_rank"] = orig_idx
            output.append(result)
        
        return output
    
    def score_pair(self, query: str, document: str) -> float:
        """
        Score a single query-document pair.
        
        Args:
            query: The search query
            document: The document text
            
        Returns:
            Relevance score (higher is more relevant)
        """
        if not self.initialized:
            return 0.5  # Neutral fallback
        
        try:
            score = self.model.predict([[query, document]])[0]
            return float(score)
        except Exception as e:
            print(f"[Reranker] Score error: {e}")
            return 0.5


class LightweightReranker:
    """
    Lightweight fallback reranker using keyword overlap and TF-IDF-like scoring.
    Used when CrossEncoder is not available.
    """
    
    def __init__(self):
        self.initialized = True
    
    def _tokenize(self, text: str) -> set:
        """Simple tokenization"""
        import re
        words = re.findall(r'\b\w+\b', text.lower())
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'can', 'of', 'to',
                     'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into'}
        return set(w for w in words if w not in stopwords and len(w) > 2)
    
    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[int, str, float]]:
        """Rerank using keyword overlap scoring"""
        query_tokens = self._tokenize(query)
        
        scored = []
        for i, doc in enumerate(documents):
            doc_tokens = self._tokenize(doc)
            
            if not query_tokens or not doc_tokens:
                overlap = 0.0
            else:
                # Jaccard-like similarity with query weighting
                intersection = len(query_tokens & doc_tokens)
                score = intersection / len(query_tokens) if query_tokens else 0
                
                # Bonus for longer matches
                if intersection >= 3:
                    score *= 1.2
                
                overlap = min(score, 1.0)
            
            scored.append((i, doc, overlap))
        
        # Sort by score
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_k]


# Create singleton - use CrossEncoder if available, else fallback
if HAS_CROSS_ENCODER:
    reranker = Reranker()
else:
    reranker = LightweightReranker()


if __name__ == "__main__":
    # Test the reranker
    query = "What is the hostel fee at UCSI?"
    
    documents = [
        "The library is open from 8am to 10pm daily.",
        "UCSI hostel accommodation costs RM 3,500 per semester for a twin room.",
        "Bus schedule: Shuttle runs every 30 minutes between campuses.",
        "Hostel facilities include WiFi, laundry, and 24-hour security.",
        "Student ID cards can be collected from the admin office.",
    ]
    
    print(f"Query: {query}\n")
    print("Reranked results:")
    
    results = reranker.rerank(query, documents, top_k=3)
    for rank, (orig_idx, doc, score) in enumerate(results, 1):
        print(f"{rank}. [Score: {score:.3f}] (was #{orig_idx+1}) {doc[:60]}...")
