"""
DangerPrecomputer module for precomputing dangerous concept sets.

This module performs background computation of all concepts that are
considered "dangerous" based on their proximity to known danger roots
in the knowledge graph.

Author: Commonsense Firewall Team
License: MIT
"""

import logging
import threading
from typing import Optional

from graph_engine import GraphEngine
from data_loader import normalize_to_uri

logger = logging.getLogger(__name__)


# Default danger root concepts - these are the starting points for danger propagation
DEFAULT_DANGER_ROOTS: list[str] = [
    "/c/en/danger",
    "/c/en/dangerous",
    "/c/en/harm",
    "/c/en/harmful",
    "/c/en/injury",
    "/c/en/death",
    "/c/en/illegal",
    "/c/en/weapon",
    "/c/en/explosion",
    "/c/en/poison",
    "/c/en/toxic",
    "/c/en/kill",
    "/c/en/murder",
    "/c/en/violence",
    "/c/en/arson",
    "/c/en/burn",
    "/c/en/electrocute",
]

# Relations that propagate danger (concepts connected via these relations inherit danger)
DANGER_PROPAGATION_RELATIONS: set[str] = {
    "Causes",
    "HasProperty",
    "UsedFor",
    "CapableOf",
    "IsA",
}


class DangerPrecomputer:
    """
    Precomputes and maintains a set of dangerous concept URIs.
    
    This class performs a BFS traversal from known danger root concepts
    to identify all concepts that are transitively related to danger.
    The computation runs in a background daemon thread to avoid blocking
    the main application startup.
    
    The danger set is computed by:
    1. Starting from a configurable set of danger root concepts
    2. Performing BFS traversal up to a configurable depth
    3. Following only relations that semantically propagate danger
    4. Storing all reached concepts in the danger_set
    
    Thread Safety:
    - The danger_set is built atomically (replaced, not modified in place)
    - The danger_set_ready flag indicates when computation is complete
    - Queries during computation return conservative results (not dangerous)
    
    Example usage:
        >>> engine = GraphEngine()
        >>> engine.build_from_loader(loader)
        >>> precomputer = DangerPrecomputer(engine)
        >>> precomputer.start_computation()
        >>> # ... later ...
        >>> if precomputer.is_dangerous_uri("/c/en/knife"):
        ...     print("Knife is dangerous!")
    """
    
    def __init__(
        self,
        graph_engine: GraphEngine,
        danger_roots: Optional[list[str]] = None,
        max_depth: int = 3,
        propagation_relations: Optional[set[str]] = None
    ):
        """
        Initialize the DangerPrecomputer.
        
        Args:
            graph_engine: The GraphEngine instance to traverse
            danger_roots: List of danger root concept URIs (uses defaults if None)
            max_depth: Maximum BFS depth for danger propagation (default 3)
            propagation_relations: Set of relations to follow (uses defaults if None)
        """
        self.graph_engine = graph_engine
        self.danger_roots = danger_roots or DEFAULT_DANGER_ROOTS
        self.max_depth = max_depth
        self.propagation_relations = propagation_relations or DANGER_PROPAGATION_RELATIONS
        
        self._danger_set: set[str] = set()
        self._danger_set_ready: bool = False
        self._computation_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
    def start_computation(self) -> None:
        """
        Start the danger set computation in a background daemon thread.
        
        This method returns immediately. Use danger_set_ready property
        to check if computation is complete, or use wait_for_completion()
        to block until done.
        """
        if self._computation_thread is not None and self._computation_thread.is_alive():
            logger.warning("Danger computation already in progress")
            return
        
        self._danger_set_ready = False
        self._computation_thread = threading.Thread(
            target=self._compute_danger_set,
            daemon=True,
            name="DangerPrecomputer"
        )
        self._computation_thread.start()
        logger.info("Started danger set computation in background thread")
    
    def _compute_danger_set(self) -> None:
        """
        Internal method that performs the actual danger set computation.
        
        This runs in a background thread and uses BFS from danger roots
        to find all transitively dangerous concepts.
        """
        try:
            logger.info(f"Computing danger set from {len(self.danger_roots)} roots, max_depth={self.max_depth}")
            
            # Use GraphEngine's BFS method for efficient traversal
            danger_set = self.graph_engine.bfs_reachable(
                sources=self.danger_roots,
                max_depth=self.max_depth,
                relation_filter=self.propagation_relations
            )
            
            # Also add concepts that CAUSE danger (reverse direction)
            # This catches things like "knife Causes injury" -> knife is dangerous
            reverse_danger = self._compute_reverse_danger()
            danger_set.update(reverse_danger)
            
            with self._lock:
                self._danger_set = danger_set
                self._danger_set_ready = True
            
            logger.info(f"Danger set computation complete: {len(danger_set)} dangerous concepts")
            
        except Exception as e:
            logger.error(f"Error computing danger set: {e}")
            with self._lock:
                self._danger_set = set()
                self._danger_set_ready = True
    
    def _compute_reverse_danger(self) -> set[str]:
        """
        Find concepts that cause or lead to danger (reverse direction).
        
        This catches patterns like:
        - knife -> Causes -> injury (knife is dangerous because it causes injury)
        - knife -> IsA -> weapon (knife is dangerous because it IS a weapon)
        - poison -> Causes -> death (poison is dangerous because it causes death)
        """
        reverse_dangerous: set[str] = set()
        
        # Relations where the head is dangerous if the tail is dangerous
        # - Causes/CapableOf: X causes danger -> X is dangerous
        # - IsA: X is a weapon -> X is dangerous
        # - HasProperty: X has property dangerous -> X is dangerous
        causal_relations = {"Causes", "CapableOf", "IsA", "HasProperty"}
        
        # First pass: direct connections to danger roots
        for root in self.danger_roots:
            root_uri = normalize_to_uri(root)
            
            # Find all concepts that have edges pointing TO this danger root
            incoming = self.graph_engine.incoming_neighbors(root_uri)
            
            for source_uri, relation, weight in incoming:
                if relation in causal_relations:
                    reverse_dangerous.add(source_uri)
        
        # Second pass: also check concepts in the danger set (from forward BFS)
        # This catches things like: knife IsA weapon, where weapon is in danger_set
        danger_categories = {normalize_to_uri(r) for r in self.danger_roots}
        
        for node in self.graph_engine.get_all_nodes():
            neighbors = self.graph_engine.neighbors(node)
            for neighbor_uri, relation, weight in neighbors:
                # If this node points to a danger category via IsA or HasProperty
                if relation in {"IsA", "HasProperty"} and neighbor_uri in danger_categories:
                    reverse_dangerous.add(node)
                # If this node Causes or is CapableOf something dangerous
                elif relation in {"Causes", "CapableOf"} and neighbor_uri in danger_categories:
                    reverse_dangerous.add(node)
        
        return reverse_dangerous
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Block until danger set computation is complete.
        
        Args:
            timeout: Maximum time to wait in seconds (None = wait forever)
            
        Returns:
            True if computation completed, False if timeout occurred
        """
        if self._computation_thread is None:
            return self._danger_set_ready
        
        self._computation_thread.join(timeout=timeout)
        return self._danger_set_ready
    
    @property
    def danger_set_ready(self) -> bool:
        """Check if the danger set computation is complete."""
        with self._lock:
            return self._danger_set_ready
    
    @property
    def danger_set(self) -> set[str]:
        """
        Get the current danger set.
        
        Note: If computation is not complete, this returns an empty set.
        Check danger_set_ready before relying on this value.
        """
        with self._lock:
            return self._danger_set.copy()
    
    def is_dangerous_uri(self, uri: str) -> bool:
        """
        Check if a concept URI is in the danger set.
        
        Args:
            uri: The concept URI to check (will be normalized)
            
        Returns:
            True if the concept is dangerous, False otherwise.
            Returns False if danger set computation is not complete.
        """
        uri = normalize_to_uri(uri)
        
        with self._lock:
            if not self._danger_set_ready:
                logger.debug(f"Danger set not ready, returning False for {uri}")
                return False
            return uri in self._danger_set
    
    def is_dangerous_concept(self, concept: str) -> bool:
        """
        Check if a concept (by name or URI) is dangerous.
        
        This is an alias for is_dangerous_uri that accepts both
        plain concept names and full URIs.
        
        Args:
            concept: The concept name or URI to check
            
        Returns:
            True if the concept is dangerous
        """
        return self.is_dangerous_uri(concept)
    
    def get_danger_path(self, uri: str, max_depth: int = 4) -> Optional[list[str]]:
        """
        Find the shortest path from a concept to a danger root.
        
        This provides an explanation for why a concept is considered dangerous.
        
        Args:
            uri: The concept URI to check
            max_depth: Maximum path length to search
            
        Returns:
            List of URIs forming the path to danger, or None if no path exists
        """
        uri = normalize_to_uri(uri)
        
        # Try to find a path to any danger root
        for root in self.danger_roots:
            root_uri = normalize_to_uri(root)
            paths = self.graph_engine.find_paths(uri, root_uri, max_depth=max_depth, max_paths=1)
            if paths:
                return paths[0]
        
        return None
    
    def get_danger_explanation(self, uri: str) -> Optional[str]:
        """
        Get a human-readable explanation of why a concept is dangerous.
        
        Args:
            uri: The concept URI to explain
            
        Returns:
            A string explanation, or None if the concept is not dangerous
        """
        if not self.is_dangerous_uri(uri):
            return None
        
        path = self.get_danger_path(uri)
        if path is None:
            return f"{uri} is in the danger set (direct match)"
        
        # Build explanation from path
        path_with_relations = self.graph_engine.find_paths_with_relations(
            path[0], path[-1], max_depth=len(path), max_paths=1
        )
        
        if not path_with_relations:
            return f"{uri} is dangerous (path: {' -> '.join(path)})"
        
        path_uris, relations, weights = path_with_relations[0]
        
        # Build human-readable explanation
        from data_loader import uri_to_label
        
        parts = []
        for i in range(len(relations)):
            head_label = uri_to_label(path_uris[i])
            tail_label = uri_to_label(path_uris[i + 1])
            rel = relations[i]
            parts.append(f"{head_label} --[{rel}]--> {tail_label}")
        
        return " | ".join(parts)
    
    def add_danger_root(self, uri: str) -> None:
        """
        Add a new danger root and recompute the danger set.
        
        Args:
            uri: The concept URI to add as a danger root
        """
        uri = normalize_to_uri(uri)
        if uri not in self.danger_roots:
            self.danger_roots.append(uri)
            logger.info(f"Added danger root: {uri}")
            self.start_computation()
    
    def get_stats(self) -> dict:
        """
        Get statistics about the danger precomputer.
        
        Returns:
            Dictionary with stats about the danger set
        """
        with self._lock:
            return {
                "ready": self._danger_set_ready,
                "danger_set_size": len(self._danger_set),
                "danger_roots_count": len(self.danger_roots),
                "max_depth": self.max_depth,
                "propagation_relations": list(self.propagation_relations),
            }
