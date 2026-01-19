"""
GraphEngine module for in-memory graph operations using NetworkX.

This module builds and maintains a directed graph representation of the
ConceptNet knowledge base, enabling efficient path finding and neighbor
queries for commonsense reasoning.

Author: Commonsense Firewall Team
License: MIT
"""

import logging
from collections import deque
from typing import Optional

import networkx as nx

from data_loader import DataLoader, normalize_to_uri

logger = logging.getLogger(__name__)


class GraphEngine:
    """
    In-memory graph engine for commonsense knowledge navigation.
    
    This class maintains a NetworkX DiGraph built from the ConceptNet
    knowledge base, providing efficient methods for:
    - Path finding between concepts (BFS with depth limit)
    - Neighbor queries with relation and weight information
    - Dynamic fact addition
    
    The graph structure:
    - Nodes: ConceptNet URIs (e.g., /c/en/knife)
    - Edges: Directed, with attributes:
        - rel: Relation type (e.g., "IsA", "Causes")
        - weight: Confidence weight (float)
    
    Example usage:
        >>> loader = DataLoader("data/conceptnet.db")
        >>> loader.ensure_bootstrap_data()
        >>> engine = GraphEngine()
        >>> engine.build_from_loader(loader)
        >>> paths = engine.find_paths("/c/en/knife", "/c/en/danger")
    """
    
    def __init__(self):
        """Initialize an empty graph engine."""
        self.graph: nx.DiGraph = nx.DiGraph()
        self._node_count: int = 0
        self._edge_count: int = 0
        
    def build_from_loader(self, data_loader: DataLoader) -> None:
        """
        Build the in-memory graph from the DataLoader's knowledge base.
        
        This method loads all edges from the SQLite database into a
        NetworkX DiGraph for fast in-memory queries.
        
        Args:
            data_loader: A DataLoader instance connected to the knowledge base
            
        Note:
            For large knowledge bases (millions of edges), this may take
            several seconds and consume significant memory. Consider
            filtering to relevant subsets for production use.
        """
        logger.info("Building graph from data loader...")
        
        edges = data_loader.get_all_edges()
        
        for head_uri, relation, tail_uri, weight in edges:
            self._add_edge_internal(head_uri, relation, tail_uri, weight)
        
        self._node_count = self.graph.number_of_nodes()
        self._edge_count = self.graph.number_of_edges()
        
        logger.info(f"Graph built: {self._node_count} nodes, {self._edge_count} edges")
    
    def _add_edge_internal(self, head_uri: str, relation: str, tail_uri: str, weight: float) -> None:
        """
        Internal method to add an edge to the graph.
        
        Handles the case where multiple edges exist between the same nodes
        by keeping the edge with the highest weight.
        """
        if self.graph.has_edge(head_uri, tail_uri):
            existing = self.graph[head_uri][tail_uri]
            if existing.get('weight', 0) < weight:
                self.graph[head_uri][tail_uri]['rel'] = relation
                self.graph[head_uri][tail_uri]['weight'] = weight
        else:
            self.graph.add_edge(head_uri, tail_uri, rel=relation, weight=weight)
    
    def add_fact(self, head_uri: str, rel: str, tail_uri: str, weight: float = 1.0) -> None:
        """
        Add a new fact (edge) to the graph.
        
        This method updates the in-memory graph immediately. Note that
        this does NOT persist to the database - use DataLoader.add_fact()
        for persistence.
        
        Args:
            head_uri: The head concept URI (will be normalized)
            rel: The relation type
            tail_uri: The tail concept URI (will be normalized)
            weight: The confidence weight (default 1.0)
        """
        head_uri = normalize_to_uri(head_uri)
        tail_uri = normalize_to_uri(tail_uri)
        
        # Normalize relation
        if rel.startswith("/r/"):
            rel = rel[3:]
        
        self._add_edge_internal(head_uri, rel, tail_uri, weight)
        self._node_count = self.graph.number_of_nodes()
        self._edge_count = self.graph.number_of_edges()
        
        logger.debug(f"Added edge: {head_uri} --[{rel}]--> {tail_uri}")
    
    def find_paths(
        self,
        source: str,
        target: str,
        max_depth: int = 4,
        max_paths: int = 10
    ) -> list[list[str]]:
        """
        Find paths between two concepts using BFS.
        
        This method performs a breadth-first search to find simple paths
        (no cycles) from source to target, limited by depth.
        
        Args:
            source: Source concept URI (will be normalized)
            target: Target concept URI (will be normalized)
            max_depth: Maximum path length (default 4)
            max_paths: Maximum number of paths to return (default 10)
            
        Returns:
            List of paths, where each path is a list of URIs from source to target
            
        Example:
            >>> engine.find_paths("/c/en/knife", "/c/en/danger")
            [['/c/en/knife', '/c/en/weapon', '/c/en/dangerous']]
        """
        source = normalize_to_uri(source)
        target = normalize_to_uri(target)
        
        if source not in self.graph or target not in self.graph:
            return []
        
        paths: list[list[str]] = []
        
        # BFS with path tracking
        # Queue contains (current_node, path_so_far)
        queue: deque[tuple[str, list[str]]] = deque([(source, [source])])
        
        while queue and len(paths) < max_paths:
            current, path = queue.popleft()
            
            if len(path) > max_depth:
                continue
            
            if current == target:
                paths.append(path)
                continue
            
            for neighbor in self.graph.successors(current):
                if neighbor not in path:  # Avoid cycles
                    new_path = path + [neighbor]
                    if len(new_path) <= max_depth + 1:
                        queue.append((neighbor, new_path))
        
        return paths
    
    def find_paths_with_relations(
        self,
        source: str,
        target: str,
        max_depth: int = 4,
        max_paths: int = 10
    ) -> list[tuple[list[str], list[str], list[float]]]:
        """
        Find paths with relation and weight information.
        
        Similar to find_paths, but returns additional information about
        the relations and weights along each path.
        
        Args:
            source: Source concept URI
            target: Target concept URI
            max_depth: Maximum path length
            max_paths: Maximum number of paths to return
            
        Returns:
            List of tuples (path_uris, relations, weights)
        """
        paths = self.find_paths(source, target, max_depth, max_paths)
        result: list[tuple[list[str], list[str], list[float]]] = []
        
        for path in paths:
            relations: list[str] = []
            weights: list[float] = []
            
            for i in range(len(path) - 1):
                edge_data = self.graph[path[i]][path[i + 1]]
                relations.append(edge_data.get('rel', 'unknown'))
                weights.append(edge_data.get('weight', 1.0))
            
            result.append((path, relations, weights))
        
        return result
    
    def neighbors(self, uri: str) -> list[tuple[str, str, float]]:
        """
        Get all outgoing neighbors of a concept with their relations and weights.
        
        Args:
            uri: The concept URI (will be normalized)
            
        Returns:
            List of tuples (neighbor_uri, relation, weight)
        """
        uri = normalize_to_uri(uri)
        
        if uri not in self.graph:
            return []
        
        result: list[tuple[str, str, float]] = []
        for neighbor in self.graph.successors(uri):
            edge_data = self.graph[uri][neighbor]
            result.append((
                neighbor,
                edge_data.get('rel', 'unknown'),
                edge_data.get('weight', 1.0)
            ))
        
        return result
    
    def incoming_neighbors(self, uri: str) -> list[tuple[str, str, float]]:
        """
        Get all incoming neighbors of a concept (concepts that point to this one).
        
        Args:
            uri: The concept URI (will be normalized)
            
        Returns:
            List of tuples (neighbor_uri, relation, weight)
        """
        uri = normalize_to_uri(uri)
        
        if uri not in self.graph:
            return []
        
        result: list[tuple[str, str, float]] = []
        for neighbor in self.graph.predecessors(uri):
            edge_data = self.graph[neighbor][uri]
            result.append((
                neighbor,
                edge_data.get('rel', 'unknown'),
                edge_data.get('weight', 1.0)
            ))
        
        return result
    
    def get_edge_data(self, source: str, target: str) -> Optional[dict]:
        """
        Get the edge data between two concepts.
        
        Args:
            source: Source concept URI
            target: Target concept URI
            
        Returns:
            Dictionary with 'rel' and 'weight' keys, or None if no edge exists
        """
        source = normalize_to_uri(source)
        target = normalize_to_uri(target)
        
        if self.graph.has_edge(source, target):
            return dict(self.graph[source][target])
        return None
    
    def has_concept(self, uri: str) -> bool:
        """Check if a concept exists in the graph."""
        return normalize_to_uri(uri) in self.graph
    
    def get_all_nodes(self) -> set[str]:
        """Get all concept URIs in the graph."""
        return set(self.graph.nodes())
    
    def get_node_degree(self, uri: str) -> tuple[int, int]:
        """
        Get the in-degree and out-degree of a concept.
        
        Args:
            uri: The concept URI
            
        Returns:
            Tuple of (in_degree, out_degree)
        """
        uri = normalize_to_uri(uri)
        if uri not in self.graph:
            return (0, 0)
        return (self.graph.in_degree(uri), self.graph.out_degree(uri))
    
    def get_high_degree_nodes(self, min_degree: int = 3) -> list[tuple[str, int]]:
        """
        Get nodes with high total degree (in + out).
        
        Useful for identifying important concepts for embedding precomputation.
        
        Args:
            min_degree: Minimum total degree to include
            
        Returns:
            List of tuples (uri, total_degree) sorted by degree descending
        """
        nodes_with_degree: list[tuple[str, int]] = []
        
        for node in self.graph.nodes():
            total_degree = self.graph.in_degree(node) + self.graph.out_degree(node)
            if total_degree >= min_degree:
                nodes_with_degree.append((node, total_degree))
        
        return sorted(nodes_with_degree, key=lambda x: x[1], reverse=True)
    
    def bfs_reachable(
        self,
        sources: list[str],
        max_depth: int = 3,
        relation_filter: Optional[set[str]] = None
    ) -> set[str]:
        """
        Find all nodes reachable from a set of source nodes via BFS.
        
        This is useful for computing danger sets or other propagation-based
        analyses.
        
        Args:
            sources: List of source concept URIs
            max_depth: Maximum BFS depth
            relation_filter: Optional set of relations to follow (if None, follow all)
            
        Returns:
            Set of all reachable node URIs (including sources)
        """
        reachable: set[str] = set()
        
        # Normalize sources
        normalized_sources = [normalize_to_uri(s) for s in sources]
        
        # Queue: (node, depth)
        queue: deque[tuple[str, int]] = deque()
        
        for source in normalized_sources:
            if source in self.graph:
                queue.append((source, 0))
                reachable.add(source)
        
        while queue:
            current, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            for neighbor in self.graph.successors(current):
                if neighbor in reachable:
                    continue
                
                # Check relation filter
                if relation_filter is not None:
                    edge_data = self.graph[current][neighbor]
                    rel = edge_data.get('rel', '')
                    if rel not in relation_filter:
                        continue
                
                reachable.add(neighbor)
                queue.append((neighbor, depth + 1))
        
        return reachable
    
    @property
    def node_count(self) -> int:
        """Get the number of nodes in the graph."""
        return self._node_count
    
    @property
    def edge_count(self) -> int:
        """Get the number of edges in the graph."""
        return self._edge_count
