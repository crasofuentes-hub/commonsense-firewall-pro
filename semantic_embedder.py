"""
SemanticEmbedder module for text embedding using sentence-transformers.

This module provides 100% offline semantic embedding capabilities using
a locally stored sentence-transformers model (all-MiniLM-L6-v2).

IMPORTANT: This module requires the model to be pre-downloaded locally.
It will NOT attempt to download models at runtime to ensure offline operation.

To download the model for offline use:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model.save('/path/to/local/model')

Author: Commonsense Firewall Team
License: MIT
"""

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ModelNotFoundError(Exception):
    """Raised when the embedding model is not found at the specified path."""
    pass


class SemanticEmbedder:
    """
    Semantic text embedder using sentence-transformers for offline operation.
    
    This class encapsulates the use of sentence-transformers/all-MiniLM-L6-v2
    for generating text embeddings. It is designed for 100% offline operation
    and will raise an error if the model is not found locally.
    
    Features:
    - Offline-only operation (no network requests)
    - Precomputed embeddings for frequent concepts
    - Efficient cosine similarity search
    - Thread-safe embedding generation
    
    Example usage:
        >>> embedder = SemanticEmbedder("/path/to/model")
        >>> embedding = embedder.encode_text("knife is dangerous")
        >>> similar = embedder.similar_concepts("sharp blade", top_k=5)
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        use_fallback: bool = True
    ):
        """
        Initialize the SemanticEmbedder with a local model.
        
        Args:
            model_path: Path to the locally stored sentence-transformers model.
                       If None, will try common locations or use fallback.
            use_fallback: If True, use a simple fallback embedder when model
                         is not available (for testing/development).
                         
        Raises:
            ModelNotFoundError: If model_path is specified but doesn't exist
                               and use_fallback is False.
        """
        self.model_path = model_path
        self.use_fallback = use_fallback
        self._model = None
        self._embedding_dim: int = 384  # all-MiniLM-L6-v2 dimension
        self._uri_to_embedding: dict[str, np.ndarray] = {}
        self._fallback_mode: bool = False
        
        self._load_model()
    
    def _load_model(self) -> None:
        """
        Load the sentence-transformers model from local storage.
        
        This method attempts to load the model from the specified path.
        If the model is not found and use_fallback is True, it falls back
        to a simple hash-based embedding for testing purposes.
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            if self.model_path and os.path.exists(self.model_path):
                logger.info(f"Loading model from local path: {self.model_path}")
                self._model = SentenceTransformer(self.model_path)
                self._embedding_dim = self._model.get_sentence_embedding_dimension()
                logger.info(f"Model loaded successfully, embedding dim: {self._embedding_dim}")
            elif self.model_path:
                if not self.use_fallback:
                    raise ModelNotFoundError(
                        f"Model not found at {self.model_path}. "
                        "Please download the model for offline use:\n"
                        "  from sentence_transformers import SentenceTransformer\n"
                        "  model = SentenceTransformer('all-MiniLM-L6-v2')\n"
                        f"  model.save('{self.model_path}')"
                    )
                logger.warning(f"Model not found at {self.model_path}, using fallback")
                self._fallback_mode = True
            else:
                # Try to load from default cache or use model name directly
                try:
                    logger.info("Attempting to load all-MiniLM-L6-v2 from cache...")
                    self._model = SentenceTransformer(
                        'all-MiniLM-L6-v2',
                        cache_folder=os.path.expanduser("~/.cache/sentence_transformers")
                    )
                    self._embedding_dim = self._model.get_sentence_embedding_dimension()
                    logger.info(f"Model loaded from cache, embedding dim: {self._embedding_dim}")
                except Exception as e:
                    if not self.use_fallback:
                        raise ModelNotFoundError(
                            f"Could not load model: {e}. "
                            "Please download the model for offline use."
                        )
                    logger.warning(f"Could not load model: {e}, using fallback")
                    self._fallback_mode = True
                    
        except ImportError:
            if not self.use_fallback:
                raise ModelNotFoundError(
                    "sentence-transformers is not installed. "
                    "Please install it: pip install sentence-transformers"
                )
            logger.warning("sentence-transformers not available, using fallback")
            self._fallback_mode = True
    
    def _fallback_encode(self, text: str) -> np.ndarray:
        """
        Fallback encoding using hash-based pseudo-embeddings.
        
        This is NOT semantically meaningful and should only be used
        for testing when the real model is not available.
        """
        # Create a deterministic pseudo-embedding based on text hash
        np.random.seed(hash(text.lower()) % (2**32))
        embedding = np.random.randn(self._embedding_dim).astype(np.float32)
        # Normalize to unit length
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a text string into a dense vector embedding.
        
        Args:
            text: The text to encode
            
        Returns:
            numpy array of shape (embedding_dim,) with the text embedding
        """
        if self._fallback_mode:
            return self._fallback_encode(text)
        
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)
    
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """
        Encode multiple texts into embeddings (batch processing).
        
        Args:
            texts: List of texts to encode
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if self._fallback_mode:
            return np.array([self._fallback_encode(t) for t in texts])
        
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.astype(np.float32)
    
    def encode_uris(self, uris: list[str]) -> dict[str, np.ndarray]:
        """
        Encode a list of concept URIs and store them for similarity search.
        
        This method extracts the concept label from each URI and encodes it.
        The embeddings are stored in the internal dictionary for later use.
        
        Args:
            uris: List of ConceptNet URIs (e.g., ["/c/en/knife", "/c/en/danger"])
            
        Returns:
            Dictionary mapping URIs to their embeddings
        """
        from data_loader import uri_to_label
        
        labels = [uri_to_label(uri) for uri in uris]
        embeddings = self.encode_texts(labels)
        
        result: dict[str, np.ndarray] = {}
        for uri, embedding in zip(uris, embeddings):
            result[uri] = embedding
            self._uri_to_embedding[uri] = embedding
        
        logger.info(f"Encoded {len(uris)} URIs, total cached: {len(self._uri_to_embedding)}")
        return result
    
    def precompute_embeddings(self, uris: list[str]) -> None:
        """
        Precompute and cache embeddings for a list of URIs.
        
        This is useful for warming up the embedder with frequently
        used concepts to speed up similarity searches.
        
        Args:
            uris: List of concept URIs to precompute
        """
        self.encode_uris(uris)
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score in range [-1, 1]
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def similar_concepts(
        self,
        text: str,
        top_k: int = 10,
        min_cosine: float = 0.5
    ) -> list[tuple[str, float]]:
        """
        Find concepts similar to the given text.
        
        This method encodes the input text and compares it against
        all precomputed URI embeddings using cosine similarity.
        
        Args:
            text: The text to find similar concepts for
            top_k: Maximum number of results to return
            min_cosine: Minimum cosine similarity threshold
            
        Returns:
            List of tuples (uri, similarity_score) sorted by score descending
        """
        if not self._uri_to_embedding:
            logger.warning("No precomputed embeddings available")
            return []
        
        text_embedding = self.encode_text(text)
        
        similarities: list[tuple[str, float]] = []
        
        for uri, uri_embedding in self._uri_to_embedding.items():
            score = self.cosine_similarity(text_embedding, uri_embedding)
            if score >= min_cosine:
                similarities.append((uri, score))
        
        # Sort by score descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def find_best_match(self, text: str, candidates: list[str]) -> Optional[tuple[str, float]]:
        """
        Find the best matching concept from a list of candidates.
        
        Args:
            text: The text to match
            candidates: List of concept URIs or labels to match against
            
        Returns:
            Tuple of (best_match, score) or None if no candidates
        """
        if not candidates:
            return None
        
        from data_loader import uri_to_label, normalize_to_uri
        
        text_embedding = self.encode_text(text)
        
        best_match: Optional[str] = None
        best_score: float = -1.0
        
        for candidate in candidates:
            # Check if we have a cached embedding
            uri = normalize_to_uri(candidate)
            if uri in self._uri_to_embedding:
                candidate_embedding = self._uri_to_embedding[uri]
            else:
                label = uri_to_label(uri)
                candidate_embedding = self.encode_text(label)
            
            score = self.cosine_similarity(text_embedding, candidate_embedding)
            if score > best_score:
                best_score = score
                best_match = uri
        
        if best_match is not None:
            return (best_match, best_score)
        return None
    
    def semantic_distance(self, text1: str, text2: str) -> float:
        """
        Compute semantic distance between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Distance score (1 - cosine_similarity), range [0, 2]
        """
        emb1 = self.encode_text(text1)
        emb2 = self.encode_text(text2)
        return 1.0 - self.cosine_similarity(emb1, emb2)
    
    def get_cached_uris(self) -> list[str]:
        """Get list of URIs with cached embeddings."""
        return list(self._uri_to_embedding.keys())
    
    def clear_cache(self) -> None:
        """Clear all cached embeddings."""
        self._uri_to_embedding.clear()
        logger.info("Embedding cache cleared")
    
    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self._embedding_dim
    
    @property
    def is_fallback_mode(self) -> bool:
        """Check if running in fallback mode (no real model)."""
        return self._fallback_mode
    
    def get_stats(self) -> dict:
        """Get statistics about the embedder."""
        return {
            "embedding_dim": self._embedding_dim,
            "cached_embeddings": len(self._uri_to_embedding),
            "fallback_mode": self._fallback_mode,
            "model_path": self.model_path,
        }
