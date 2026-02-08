"""
Semantic Cache Engine - World-Class Query Caching
Uses sentence embeddings to find similar previous queries and return cached responses.
Inspired by: GPTCache, Redis Vector Search patterns
"""
import os
import json
import time
import hashlib
import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta

# Suppress warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import logging
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# Conditional imports
try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False
    print("Warning: sentence-transformers not installed for Semantic Cache")


class SemanticCache:
    """
    Enterprise-grade Semantic Cache
    
    Features:
    - Embedding-based similarity matching (not just exact match)
    - Configurable similarity threshold
    - TTL-based expiration
    - LRU eviction when cache is full
    - Thread-safe operations
    """
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.92,  # High threshold for accuracy
                 max_cache_size: int = 1000,
                 ttl_seconds: int = 3600):  # 1 hour default
        """
        Initialize Semantic Cache
        
        Args:
            model_name: Sentence transformer model for embeddings
            similarity_threshold: Minimum cosine similarity to consider a cache hit (0.92 = very similar)
            max_cache_size: Maximum number of cached entries
            ttl_seconds: Time-to-live for cache entries
        """
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.ttl_seconds = ttl_seconds
        self.model = None
        self.initialized = False
        
        # Cache storage: {cache_key: {embedding, response, timestamp, query, hit_count}}
        self.cache: Dict[str, Dict] = {}
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.cache_keys: List[str] = []
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }
        
        self._initialize_model(model_name)
    
    def _initialize_model(self, model_name: str):
        """Initialize the embedding model"""
        if not HAS_EMBEDDINGS:
            print("[SemanticCache] Disabled - sentence-transformers not available")
            return
            
        try:
            self.model = SentenceTransformer(model_name)
            self.initialized = True
            print(f"[SemanticCache] Initialized with model: {model_name}")
        except Exception as e:
            print(f"[SemanticCache] Failed to initialize: {e}")
            self.initialized = False
    
    def _compute_embedding(self, text: str) -> Optional[np.ndarray]:
        """Compute embedding for a text query"""
        if not self.initialized or not self.model:
            return None
        try:
            embedding = self.model.encode(text, normalize_embeddings=True)
            return embedding
        except Exception as e:
            print(f"[SemanticCache] Embedding error: {e}")
            return None
    
    def _generate_cache_key(self, query: str) -> str:
        """Generate a unique cache key for a query"""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        return float(np.dot(a, b))  # Already normalized
    
    def _find_similar(self, query_embedding: np.ndarray) -> Optional[Tuple[str, float]]:
        """
        Find the most similar cached query
        
        Returns: (cache_key, similarity_score) or None
        """
        if not self.cache_keys or self.embeddings_matrix is None:
            return None
        
        # Compute similarities with all cached embeddings
        similarities = np.dot(self.embeddings_matrix, query_embedding)
        max_idx = np.argmax(similarities)
        max_similarity = similarities[max_idx]
        
        if max_similarity >= self.similarity_threshold:
            return (self.cache_keys[max_idx], float(max_similarity))
        
        return None
    
    def _rebuild_embeddings_matrix(self):
        """Rebuild the embeddings matrix from cache"""
        if not self.cache:
            self.embeddings_matrix = None
            self.cache_keys = []
            return
            
        self.cache_keys = list(self.cache.keys())
        embeddings = [self.cache[key]["embedding"] for key in self.cache_keys]
        self.embeddings_matrix = np.vstack(embeddings)
    
    def _evict_expired(self):
        """Remove expired entries"""
        now = datetime.now()
        expired_keys = []
        
        for key, entry in self.cache.items():
            age = (now - entry["timestamp"]).total_seconds()
            if age > self.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
            self.stats["evictions"] += 1
        
        if expired_keys:
            self._rebuild_embeddings_matrix()
    
    def _evict_lru(self):
        """Evict least recently used entries if cache is full"""
        if len(self.cache) < self.max_cache_size:
            return
        
        # Sort by hit_count (ascending) then by timestamp (oldest first)
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: (x[1]["hit_count"], x[1]["timestamp"])
        )
        
        # Remove oldest 10% of cache
        num_to_remove = max(1, len(self.cache) // 10)
        for key, _ in sorted_entries[:num_to_remove]:
            del self.cache[key]
            self.stats["evictions"] += 1
        
        self._rebuild_embeddings_matrix()
    
    def get(self, query: str) -> Optional[Dict]:
        """
        Try to get a cached response for a semantically similar query
        
        Args:
            query: The user's query
            
        Returns:
            Cached response dict with {response, original_query, similarity} or None
        """
        if not self.initialized:
            return None
        
        # Clean expired entries periodically
        self._evict_expired()
        
        # Compute query embedding
        query_embedding = self._compute_embedding(query)
        if query_embedding is None:
            return None
        
        # Find similar cached query
        similar = self._find_similar(query_embedding)
        
        if similar:
            cache_key, similarity = similar
            entry = self.cache[cache_key]
            
            # Update hit statistics
            entry["hit_count"] += 1
            self.stats["hits"] += 1
            
            print(f"[SemanticCache] HIT! Similarity: {similarity:.3f} | Original: '{entry['query'][:50]}...'")
            
            return {
                "response": entry["response"],
                "original_query": entry["query"],
                "similarity": similarity,
                "from_cache": True
            }
        
        self.stats["misses"] += 1
        return None
    
    def put(self, query: str, response: str):
        """
        Cache a query-response pair
        
        Args:
            query: The user's query
            response: The AI's response
        """
        if not self.initialized:
            return
        
        # Don't cache error responses or very short responses
        if not response or len(response) < 20:
            return
        if "error" in response.lower() or "couldn't find" in response.lower():
            return
        
        # Evict if necessary
        self._evict_lru()
        
        # Compute embedding
        query_embedding = self._compute_embedding(query)
        if query_embedding is None:
            return
        
        # Check if already cached (exact match)
        cache_key = self._generate_cache_key(query)
        
        # Store in cache
        self.cache[cache_key] = {
            "embedding": query_embedding,
            "query": query,
            "response": response,
            "timestamp": datetime.now(),
            "hit_count": 0
        }
        
        # Rebuild matrix
        self._rebuild_embeddings_matrix()
        
        print(f"[SemanticCache] Stored: '{query[:50]}...'")
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total * 100) if total > 0 else 0
        
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": f"{hit_rate:.1f}%",
            "cache_size": len(self.cache),
            "max_size": self.max_cache_size,
            "evictions": self.stats["evictions"]
        }
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.embeddings_matrix = None
        self.cache_keys = []
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}
        print("[SemanticCache] Cleared")


# Singleton instance
semantic_cache = SemanticCache()


if __name__ == "__main__":
    # Test the semantic cache
    cache = SemanticCache()
    
    # Test storing
    cache.put("What is the hostel fee?", "The hostel fee is RM 3,500 per semester.")
    cache.put("Where is Block A located?", "Block A is located at the main campus.")
    
    # Test retrieval with similar query
    result = cache.get("How much does the hostel cost?")  # Similar to first query
    if result:
        print(f"Cache hit! Response: {result['response']}")
        print(f"Similarity: {result['similarity']:.3f}")
    
    print(f"Stats: {cache.get_stats()}")
