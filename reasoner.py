"""
Reasoner module for hybrid symbolic-semantic commonsense reasoning.

This module implements the core reasoning capabilities that combine
graph-based symbolic reasoning with semantic embeddings to:
- Extract concepts from text
- Find inference chains in the knowledge graph
- Detect dangerous content
- Check for logical contradictions

Author: Commonsense Firewall Team
License: MIT
"""

import logging
import re
from typing import Optional

from data_loader import normalize_to_uri, uri_to_label
from graph_engine import GraphEngine
from semantic_embedder import SemanticEmbedder
from danger_precomputer import DangerPrecomputer

logger = logging.getLogger(__name__)


# Common stop words to filter out during concept extraction
STOP_WORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "and", "but", "if", "or",
    "because", "until", "while", "although", "though", "after", "before",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "about", "against", "both", "any",
}

# Relations that indicate contradiction
CONTRADICTION_RELATIONS: set[str] = {"Antonym", "NotCapableOf", "DistinctFrom"}

# Relations that indicate properties
PROPERTY_RELATIONS: set[str] = {"HasProperty", "IsA"}


class Reasoner:
    """
    Hybrid symbolic-semantic reasoner for commonsense inference.
    
    This class combines graph-based symbolic reasoning with semantic
    embeddings to perform various reasoning tasks:
    
    1. Concept Extraction: Extract relevant concepts from text
    2. Inference Generation: Find reasoning chains in the knowledge graph
    3. Danger Detection: Identify dangerous content using the danger set
    4. Consistency Checking: Detect logical contradictions
    
    The reasoner uses a two-stage approach:
    1. Symbolic matching: Try to map text tokens to known concepts
    2. Semantic fallback: Use embeddings to find similar concepts
    
    Example usage:
        >>> reasoner = Reasoner(graph_engine, embedder)
        >>> inferences = reasoner.infer_from_text("knife is sharp")
        >>> dangers = reasoner.check_danger_for_text("how to use a gun")
    """
    
    def __init__(
        self,
        graph_engine: GraphEngine,
        embedder: SemanticEmbedder,
        semantic_threshold: float = 0.7
    ):
        """
        Initialize the Reasoner.
        
        Args:
            graph_engine: GraphEngine instance for symbolic reasoning
            embedder: SemanticEmbedder instance for semantic matching
            semantic_threshold: Minimum cosine similarity for semantic matches
        """
        self.graph_engine = graph_engine
        self.embedder = embedder
        self.semantic_threshold = semantic_threshold
    
    def _simple_lemmatize(self, word: str) -> str:
        """
        Simple heuristic lemmatization without external libraries.
        
        This handles common English suffixes to normalize words.
        For production use, consider using spaCy or NLTK.
        """
        word = word.lower().strip()
        
        # Common suffix rules
        if word.endswith("ies") and len(word) > 4:
            return word[:-3] + "y"
        if word.endswith("es") and len(word) > 3:
            if word.endswith("sses") or word.endswith("xes") or word.endswith("ches") or word.endswith("shes"):
                return word[:-2]
            return word[:-1]
        if word.endswith("s") and len(word) > 2 and not word.endswith("ss"):
            return word[:-1]
        if word.endswith("ing") and len(word) > 5:
            base = word[:-3]
            if base.endswith("e"):
                return base
            return base
        if word.endswith("ed") and len(word) > 4:
            base = word[:-2]
            if base.endswith("i"):
                return base[:-1] + "y"
            return base
        if word.endswith("ly") and len(word) > 4:
            return word[:-2]
        
        return word
    
    def _tokenize_text(self, text: str) -> list[str]:
        """
        Tokenize text into words, filtering stop words and normalizing.
        
        Args:
            text: Input text
            
        Returns:
            List of normalized tokens
        """
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        
        # Filter and normalize
        result: list[str] = []
        for token in tokens:
            if token in STOP_WORDS:
                continue
            if len(token) < 2:
                continue
            if token.isdigit():
                continue
            
            lemma = self._simple_lemmatize(token)
            if lemma and lemma not in STOP_WORDS:
                result.append(lemma)
        
        return result
    
    def _extract_ngrams(self, tokens: list[str], max_n: int = 3) -> list[str]:
        """
        Extract n-grams from tokens for multi-word concept matching.
        
        Args:
            tokens: List of tokens
            max_n: Maximum n-gram size
            
        Returns:
            List of n-grams (including unigrams)
        """
        ngrams: list[str] = []
        
        for n in range(max_n, 0, -1):
            for i in range(len(tokens) - n + 1):
                ngram = "_".join(tokens[i:i + n])
                ngrams.append(ngram)
        
        return ngrams
    
    def extract_concepts(self, text: str) -> list[tuple[str, float]]:
        """
        Extract concepts from text with confidence scores.
        
        This method uses a two-stage approach:
        1. Try to match tokens/n-grams to known concepts in the graph
        2. Use semantic similarity to find related concepts
        
        Args:
            text: Input text
            
        Returns:
            List of tuples (concept_uri, confidence_score)
        """
        tokens = self._tokenize_text(text)
        ngrams = self._extract_ngrams(tokens)
        
        concepts: list[tuple[str, float]] = []
        matched_tokens: set[str] = set()
        
        # Stage 1: Exact matching
        for ngram in ngrams:
            uri = normalize_to_uri(ngram)
            if self.graph_engine.has_concept(uri):
                # Check if any component token was already matched
                component_tokens = set(ngram.split("_"))
                if not component_tokens.intersection(matched_tokens):
                    concepts.append((uri, 1.0))
                    matched_tokens.update(component_tokens)
        
        # Stage 2: Semantic matching for unmatched tokens
        unmatched = [t for t in tokens if t not in matched_tokens]
        
        if unmatched and self.embedder.get_cached_uris():
            for token in unmatched:
                similar = self.embedder.similar_concepts(
                    token,
                    top_k=3,
                    min_cosine=self.semantic_threshold
                )
                for uri, score in similar:
                    if uri not in [c[0] for c in concepts]:
                        concepts.append((uri, score))
        
        # Sort by confidence
        concepts.sort(key=lambda x: x[1], reverse=True)
        
        return concepts
    
    def score_inference(
        self,
        path: list[str],
        edge_weights: list[float],
        query_text: str
    ) -> float:
        """
        Score an inference path based on multiple factors.
        
        The score combines:
        - Average edge weight (higher is better)
        - Path length penalty (shorter is better)
        - Semantic relevance to the query
        
        Args:
            path: List of concept URIs in the path
            edge_weights: List of edge weights along the path
            query_text: Original query text for semantic relevance
            
        Returns:
            Combined score in range [0, 1]
        """
        if not path or not edge_weights:
            return 0.0
        
        # Component 1: Average edge weight (normalized to [0, 1])
        avg_weight = sum(edge_weights) / len(edge_weights)
        weight_score = min(avg_weight / 3.0, 1.0)  # Assume max weight ~3
        
        # Component 2: Path length penalty
        # Shorter paths are better (1 edge = 1.0, 4 edges = 0.25)
        length_score = 1.0 / len(edge_weights)
        
        # Component 3: Semantic relevance
        # Average similarity between query and path concepts
        semantic_scores: list[float] = []
        query_embedding = self.embedder.encode_text(query_text)
        
        for uri in path:
            label = uri_to_label(uri)
            concept_embedding = self.embedder.encode_text(label)
            sim = self.embedder.cosine_similarity(query_embedding, concept_embedding)
            semantic_scores.append(max(0, sim))
        
        semantic_score = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0.0
        
        # Combine scores with weights
        combined = (
            0.3 * weight_score +
            0.3 * length_score +
            0.4 * semantic_score
        )
        
        return combined
    
    def infer_from_text(
        self,
        text: str,
        max_inferences: int = 10,
        max_depth: int = 4
    ) -> list[dict]:
        """
        Generate inferences from text using the knowledge graph.
        
        This method extracts concepts from the text and finds
        inference chains connecting them in the knowledge graph.
        
        Args:
            text: Input text
            max_inferences: Maximum number of inferences to return
            max_depth: Maximum path depth for inference chains
            
        Returns:
            List of inference dictionaries with:
            - path_uris: List of concept URIs in the path
            - path_relations: List of relations along the path
            - score: Combined inference score
            - explanation: Human-readable explanation
        """
        concepts = self.extract_concepts(text)
        
        if not concepts:
            logger.debug(f"No concepts extracted from: {text}")
            return []
        
        inferences: list[dict] = []
        seen_paths: set[tuple] = set()
        
        # Find paths between pairs of concepts
        for i, (uri1, conf1) in enumerate(concepts):
            # Also explore neighbors of each concept
            neighbors = self.graph_engine.neighbors(uri1)
            
            for neighbor_uri, relation, weight in neighbors:
                path_key = (uri1, neighbor_uri)
                if path_key in seen_paths:
                    continue
                seen_paths.add(path_key)
                
                # Create inference for direct neighbor
                path_uris = [uri1, neighbor_uri]
                path_relations = [relation]
                edge_weights = [weight]
                
                score = self.score_inference(path_uris, edge_weights, text)
                
                explanation = self._build_explanation(path_uris, path_relations)
                
                inferences.append({
                    "path_uris": path_uris,
                    "path_relations": path_relations,
                    "score": score,
                    "explanation": explanation,
                })
            
            # Find paths between concept pairs
            for j, (uri2, conf2) in enumerate(concepts):
                if i >= j:
                    continue
                
                paths_with_rels = self.graph_engine.find_paths_with_relations(
                    uri1, uri2, max_depth=max_depth, max_paths=3
                )
                
                for path_uris, path_relations, edge_weights in paths_with_rels:
                    path_key = tuple(path_uris)
                    if path_key in seen_paths:
                        continue
                    seen_paths.add(path_key)
                    
                    score = self.score_inference(path_uris, edge_weights, text)
                    explanation = self._build_explanation(path_uris, path_relations)
                    
                    inferences.append({
                        "path_uris": path_uris,
                        "path_relations": path_relations,
                        "score": score,
                        "explanation": explanation,
                    })
        
        # Sort by score and limit
        inferences.sort(key=lambda x: x["score"], reverse=True)
        return inferences[:max_inferences]
    
    def _build_explanation(self, path_uris: list[str], relations: list[str]) -> str:
        """
        Build a human-readable explanation from a path.
        
        Args:
            path_uris: List of concept URIs
            relations: List of relations between concepts
            
        Returns:
            Human-readable explanation string
        """
        parts: list[str] = []
        
        for i in range(len(relations)):
            head = uri_to_label(path_uris[i])
            tail = uri_to_label(path_uris[i + 1])
            rel = relations[i]
            parts.append(f"{head} --[{rel}]--> {tail}")
        
        return " | ".join(parts)
    
    def check_danger_for_text(
        self,
        text: str,
        danger_precomputer: DangerPrecomputer,
        max_depth: int = 3
    ) -> list[dict]:
        """
        Check text for dangerous content.
        
        This method extracts concepts from the text and checks if any
        are in the danger set or have short paths to dangerous concepts.
        
        Args:
            text: Input text to check
            danger_precomputer: DangerPrecomputer instance
            max_depth: Maximum path depth for danger path search
            
        Returns:
            List of danger findings with:
            - concept_uri: The dangerous concept found
            - concept_label: Human-readable label
            - is_direct: True if concept is directly in danger set
            - danger_path: Path to danger root (if applicable)
            - explanation: Human-readable explanation
        """
        concepts = self.extract_concepts(text)
        findings: list[dict] = []
        
        for uri, confidence in concepts:
            # Check if directly in danger set
            if danger_precomputer.is_dangerous_uri(uri):
                explanation = danger_precomputer.get_danger_explanation(uri)
                findings.append({
                    "concept_uri": uri,
                    "concept_label": uri_to_label(uri),
                    "confidence": confidence,
                    "is_direct": True,
                    "danger_path": None,
                    "explanation": explanation or f"{uri_to_label(uri)} is in the danger set",
                })
            else:
                # Check for short paths to danger
                danger_path = danger_precomputer.get_danger_path(uri, max_depth=max_depth)
                if danger_path:
                    path_labels = [uri_to_label(u) for u in danger_path]
                    findings.append({
                        "concept_uri": uri,
                        "concept_label": uri_to_label(uri),
                        "confidence": confidence,
                        "is_direct": False,
                        "danger_path": danger_path,
                        "explanation": f"{uri_to_label(uri)} leads to danger via: {' -> '.join(path_labels)}",
                    })
        
        return findings
    
    def check_basic_consistency(self, text: str) -> list[dict]:
        """
        Check text for basic logical contradictions.
        
        This method detects contradictions using ConceptNet relations:
        - Antonym: "water is dry" contradicts "water HasProperty wet" + "wet Antonym dry"
        - NotCapableOf: "fish can fly" contradicts "fish NotCapableOf fly"
        - DistinctFrom: Claiming two distinct things are the same
        
        Args:
            text: Input text to check
            
        Returns:
            List of contradiction findings with:
            - type: Type of contradiction
            - concepts: Concepts involved
            - explanation: Human-readable explanation
        """
        concepts = self.extract_concepts(text)
        contradictions: list[dict] = []
        
        if len(concepts) < 2:
            return contradictions
        
        # Get all concept URIs
        concept_uris = [uri for uri, _ in concepts]
        
        # Check for antonym contradictions
        # Pattern: text mentions X and Y where X HasProperty P and Y HasProperty Q and P Antonym Q
        for i, uri1 in enumerate(concept_uris):
            props1 = self._get_properties(uri1)
            
            for j, uri2 in enumerate(concept_uris):
                if i >= j:
                    continue
                
                props2 = self._get_properties(uri2)
                
                # Check if any properties are antonyms
                for prop1 in props1:
                    for prop2 in props2:
                        if self._are_antonyms(prop1, prop2):
                            contradictions.append({
                                "type": "property_antonym",
                                "concepts": [uri1, uri2],
                                "properties": [prop1, prop2],
                                "explanation": (
                                    f"Contradiction: {uri_to_label(uri1)} has property "
                                    f"'{uri_to_label(prop1)}' which is antonym of "
                                    f"'{uri_to_label(prop2)}' (property of {uri_to_label(uri2)})"
                                ),
                            })
        
        # Check for direct antonym mentions
        for i, uri1 in enumerate(concept_uris):
            for j, uri2 in enumerate(concept_uris):
                if i >= j:
                    continue
                
                if self._are_antonyms(uri1, uri2):
                    contradictions.append({
                        "type": "direct_antonym",
                        "concepts": [uri1, uri2],
                        "explanation": (
                            f"Potential contradiction: '{uri_to_label(uri1)}' and "
                            f"'{uri_to_label(uri2)}' are antonyms"
                        ),
                    })
        
        # Check for property contradictions (e.g., "water is dry")
        # This looks for patterns where a concept is claimed to have a property
        # that contradicts its known properties
        tokens = self._tokenize_text(text)
        
        for concept_uri, _ in concepts:
            known_props = self._get_properties(concept_uri)
            
            for token in tokens:
                token_uri = normalize_to_uri(token)
                
                # Check if token is antonym of any known property
                for prop in known_props:
                    if self._are_antonyms(token_uri, prop):
                        contradictions.append({
                            "type": "property_contradiction",
                            "concepts": [concept_uri],
                            "claimed_property": token_uri,
                            "known_property": prop,
                            "explanation": (
                                f"Contradiction: {uri_to_label(concept_uri)} is claimed to be "
                                f"'{token}' but is known to have property '{uri_to_label(prop)}' "
                                f"(antonym of '{token}')"
                            ),
                        })
        
        return contradictions
    
    def _get_properties(self, uri: str) -> list[str]:
        """
        Get all properties of a concept.
        
        Args:
            uri: Concept URI
            
        Returns:
            List of property URIs
        """
        properties: list[str] = []
        
        neighbors = self.graph_engine.neighbors(uri)
        for neighbor_uri, relation, weight in neighbors:
            if relation in PROPERTY_RELATIONS:
                properties.append(neighbor_uri)
        
        return properties
    
    def _are_antonyms(self, uri1: str, uri2: str) -> bool:
        """
        Check if two concepts are antonyms.
        
        Args:
            uri1: First concept URI
            uri2: Second concept URI
            
        Returns:
            True if the concepts are antonyms
        """
        uri1 = normalize_to_uri(uri1)
        uri2 = normalize_to_uri(uri2)
        
        # Check direct edge
        edge_data = self.graph_engine.get_edge_data(uri1, uri2)
        if edge_data and edge_data.get("rel") in CONTRADICTION_RELATIONS:
            return True
        
        # Check reverse edge
        edge_data = self.graph_engine.get_edge_data(uri2, uri1)
        if edge_data and edge_data.get("rel") in CONTRADICTION_RELATIONS:
            return True
        
        return False
    
    def analyze_text(
        self,
        text: str,
        danger_precomputer: Optional[DangerPrecomputer] = None
    ) -> dict:
        """
        Perform comprehensive analysis of text.
        
        This method combines all reasoning capabilities:
        - Concept extraction
        - Inference generation
        - Danger detection (if precomputer provided)
        - Consistency checking
        
        Args:
            text: Input text to analyze
            danger_precomputer: Optional DangerPrecomputer for danger detection
            
        Returns:
            Dictionary with all analysis results
        """
        result = {
            "text": text,
            "concepts": self.extract_concepts(text),
            "inferences": self.infer_from_text(text),
            "contradictions": self.check_basic_consistency(text),
            "dangers": [],
        }
        
        if danger_precomputer:
            result["dangers"] = self.check_danger_for_text(text, danger_precomputer)
        
        return result
