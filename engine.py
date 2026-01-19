"""
FastCommonsenseEngine - Main facade for the commonsense reasoning firewall.

This module provides the primary interface for using the commonsense
reasoning engine as a logical firewall for LLM responses. It integrates
all components and provides:

- High-level API for querying and verifying text
- Rate limiting to prevent abuse
- Circuit breaker pattern for fault tolerance
- Latency monitoring and logging
- CLI interface for interactive use

USAGE AS LLM FIREWALL:
======================
The primary use case is to validate LLM responses before showing them to users:

    engine = FastCommonsenseEngine()
    
    # Before showing LLM response to user:
    is_safe, reason = engine.verify_response(llm_response)
    if not is_safe:
        # Block or flag the response
        log_unsafe_response(llm_response, reason)
        return fallback_response()
    
    return llm_response

EXPANDING THE KNOWLEDGE BASE:
=============================
To use full ConceptNet instead of the bootstrap mini-dataset:

1. Download ConceptNet assertions:
   wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz

2. Filter and import (see data_loader.py for detailed instructions):
   - Filter for English, weight > 1.0, relevant relations
   - Use DataLoader.add_fact() to populate the database

3. For ATOMIC integration:
   - Download ATOMIC 2020 dataset
   - Normalize to (head, relation, tail) format
   - Import using DataLoader.add_fact()

Author: Commonsense Firewall Team
License: MIT
"""

import argparse
import logging
import threading
import time
from collections import deque
from enum import Enum
from functools import lru_cache
from typing import Optional

from data_loader import DataLoader, uri_to_label
from graph_engine import GraphEngine
from danger_precomputer import DangerPrecomputer
from semantic_embedder import SemanticEmbedder
from reasoner import Reasoner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    pass


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""
    pass


class FastCommonsenseEngine:
    """
    Main facade for the commonsense reasoning firewall.
    
    This class integrates all components of the reasoning engine and
    provides a high-level API for:
    
    1. Querying commonsense knowledge
    2. Checking if concepts are dangerous
    3. Verifying LLM responses for safety and consistency
    
    Features:
    - Rate limiting (configurable requests per second)
    - Circuit breaker pattern for fault tolerance
    - Latency monitoring and logging
    - LRU caching for repeated queries
    
    Example usage:
        >>> engine = FastCommonsenseEngine()
        >>> 
        >>> # Query commonsense
        >>> inferences = engine.query_commonsense("knife is sharp")
        >>> 
        >>> # Check danger
        >>> is_dangerous, explanation = engine.is_dangerous("how to make a bomb")
        >>> 
        >>> # Verify LLM response
        >>> is_safe, reason = engine.verify_response("Use a knife to cut vegetables")
    """
    
    def __init__(
        self,
        db_path: str = "data/conceptnet.db",
        model_path: Optional[str] = None,
        max_search_depth: int = 4,
        danger_depth: int = 3,
        semantic_threshold: float = 0.7,
        rate_limit_per_second: int = 50,
        circuit_breaker_max_failures: int = 5,
        circuit_breaker_timeout: float = 60.0,
        use_fallback_embedder: bool = True
    ):
        """
        Initialize the FastCommonsenseEngine.
        
        Args:
            db_path: Path to the ConceptNet SQLite database
            model_path: Path to the sentence-transformers model (optional)
            max_search_depth: Maximum depth for path searches
            danger_depth: Maximum depth for danger propagation
            semantic_threshold: Minimum cosine similarity for semantic matches
            rate_limit_per_second: Maximum requests per second
            circuit_breaker_max_failures: Failures before opening circuit
            circuit_breaker_timeout: Seconds before trying to close circuit
            use_fallback_embedder: Use fallback embedder if model not found
        """
        self.db_path = db_path
        self.model_path = model_path
        self.max_search_depth = max_search_depth
        self.danger_depth = danger_depth
        self.semantic_threshold = semantic_threshold
        
        # Rate limiting
        self._rate_limit = rate_limit_per_second
        self._request_times: deque = deque(maxlen=rate_limit_per_second * 2)
        self._rate_lock = threading.Lock()
        
        # Circuit breaker
        self._circuit_state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._max_failures = circuit_breaker_max_failures
        self._circuit_timeout = circuit_breaker_timeout
        self._last_failure_time: Optional[float] = None
        self._circuit_lock = threading.Lock()
        self._half_open_successes = 0
        self._half_open_threshold = 3  # Successes needed to close circuit
        
        # Initialize components
        logger.info("Initializing FastCommonsenseEngine...")
        self._init_components(use_fallback_embedder)
        logger.info("FastCommonsenseEngine initialized successfully")
    
    def _init_components(self, use_fallback_embedder: bool) -> None:
        """Initialize all engine components."""
        # Data loader
        logger.info(f"Initializing DataLoader with db_path={self.db_path}")
        self.data_loader = DataLoader(self.db_path)
        self.data_loader.ensure_bootstrap_data()
        
        # Graph engine
        logger.info("Building graph from data loader...")
        self.graph_engine = GraphEngine()
        self.graph_engine.build_from_loader(self.data_loader)
        
        # Semantic embedder
        logger.info("Initializing SemanticEmbedder...")
        self.embedder = SemanticEmbedder(
            model_path=self.model_path,
            use_fallback=use_fallback_embedder
        )
        
        # Precompute embeddings for graph nodes
        self._precompute_embeddings()
        
        # Danger precomputer (runs in background thread)
        logger.info("Starting danger precomputation...")
        self.danger_precomputer = DangerPrecomputer(
            self.graph_engine,
            max_depth=self.danger_depth
        )
        self.danger_precomputer.start_computation()
        
        # Reasoner
        logger.info("Initializing Reasoner...")
        self.reasoner = Reasoner(
            self.graph_engine,
            self.embedder,
            semantic_threshold=self.semantic_threshold
        )
    
    def _precompute_embeddings(self) -> None:
        """Precompute embeddings for frequent graph nodes."""
        # Get high-degree nodes (important concepts)
        high_degree_nodes = self.graph_engine.get_high_degree_nodes(min_degree=2)
        
        # Limit to top 100 for efficiency
        nodes_to_embed = [uri for uri, _ in high_degree_nodes[:100]]
        
        if nodes_to_embed:
            logger.info(f"Precomputing embeddings for {len(nodes_to_embed)} nodes...")
            self.embedder.precompute_embeddings(nodes_to_embed)
    
    def _check_rate_limit(self) -> None:
        """
        Check and enforce rate limiting.
        
        Raises:
            RateLimitExceeded: If rate limit is exceeded
        """
        current_time = time.time()
        
        with self._rate_lock:
            # Remove old timestamps (older than 1 second)
            while self._request_times and current_time - self._request_times[0] > 1.0:
                self._request_times.popleft()
            
            # Check if limit exceeded
            if len(self._request_times) >= self._rate_limit:
                raise RateLimitExceeded(
                    f"Rate limit exceeded: {self._rate_limit} requests per second"
                )
            
            # Record this request
            self._request_times.append(current_time)
    
    def _check_circuit_breaker(self) -> None:
        """
        Check circuit breaker state.
        
        Raises:
            CircuitBreakerOpen: If circuit is open
        """
        with self._circuit_lock:
            if self._circuit_state == CircuitState.OPEN:
                # Check if timeout has passed
                if self._last_failure_time is not None:
                    elapsed = time.time() - self._last_failure_time
                    if elapsed >= self._circuit_timeout:
                        logger.info("Circuit breaker transitioning to HALF_OPEN")
                        self._circuit_state = CircuitState.HALF_OPEN
                        self._half_open_successes = 0
                    else:
                        raise CircuitBreakerOpen(
                            f"Circuit breaker is OPEN. Retry in {self._circuit_timeout - elapsed:.1f}s"
                        )
                else:
                    raise CircuitBreakerOpen("Circuit breaker is OPEN")
    
    def _record_success(self) -> None:
        """Record a successful operation for circuit breaker."""
        with self._circuit_lock:
            if self._circuit_state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self._half_open_threshold:
                    logger.info("Circuit breaker transitioning to CLOSED")
                    self._circuit_state = CircuitState.CLOSED
                    self._consecutive_failures = 0
            elif self._circuit_state == CircuitState.CLOSED:
                self._consecutive_failures = 0
    
    def _record_failure(self) -> None:
        """Record a failed operation for circuit breaker."""
        with self._circuit_lock:
            self._consecutive_failures += 1
            self._last_failure_time = time.time()
            
            if self._circuit_state == CircuitState.HALF_OPEN:
                logger.warning("Failure in HALF_OPEN state, returning to OPEN")
                self._circuit_state = CircuitState.OPEN
                # Exponential backoff
                self._circuit_timeout = min(self._circuit_timeout * 2, 300)
            elif self._consecutive_failures >= self._max_failures:
                logger.warning(
                    f"Circuit breaker OPEN after {self._consecutive_failures} failures"
                )
                self._circuit_state = CircuitState.OPEN
    
    def _measure_latency(self, func_name: str, start_time: float) -> float:
        """Log and return latency for an operation."""
        latency = (time.perf_counter() - start_time) * 1000  # Convert to ms
        logger.info(f"{func_name} latency: {latency:.2f}ms")
        return latency
    
    def query_commonsense(self, concept_or_text: str) -> list[dict]:
        """
        Query commonsense knowledge about a concept or text.
        
        This method extracts concepts from the input and finds
        relevant inferences in the knowledge graph.
        
        Args:
            concept_or_text: A concept name or text to analyze
            
        Returns:
            List of inference dictionaries with path_uris, path_relations,
            score, and explanation
            
        Raises:
            RateLimitExceeded: If rate limit is exceeded
            CircuitBreakerOpen: If circuit breaker is open
        """
        start_time = time.perf_counter()
        
        try:
            self._check_rate_limit()
            self._check_circuit_breaker()
            
            result = self._query_commonsense_cached(concept_or_text)
            
            self._record_success()
            self._measure_latency("query_commonsense", start_time)
            
            return result
            
        except (RateLimitExceeded, CircuitBreakerOpen):
            raise
        except Exception as e:
            self._record_failure()
            logger.error(f"Error in query_commonsense: {e}")
            raise
    
    @lru_cache(maxsize=4096)
    def _query_commonsense_cached(self, concept_or_text: str) -> tuple:
        """Cached implementation of query_commonsense."""
        inferences = self.reasoner.infer_from_text(
            concept_or_text,
            max_depth=self.max_search_depth
        )
        # Convert to tuple for caching (lists aren't hashable)
        return tuple(
            tuple(sorted(inf.items())) for inf in inferences
        )
    
    def is_dangerous(self, concept_or_text: str) -> tuple[bool, Optional[list[str]]]:
        """
        Check if a concept or text is dangerous.
        
        This method normalizes the input to concept URIs and checks
        against the precomputed danger set.
        
        Args:
            concept_or_text: A concept name or text to check
            
        Returns:
            Tuple of (is_dangerous: bool, explanation: list[str] or None)
            The explanation contains the path to danger if found.
            
        Raises:
            RateLimitExceeded: If rate limit is exceeded
            CircuitBreakerOpen: If circuit breaker is open
        """
        start_time = time.perf_counter()
        
        try:
            self._check_rate_limit()
            self._check_circuit_breaker()
            
            # Wait for danger set if not ready (with timeout)
            if not self.danger_precomputer.danger_set_ready:
                self.danger_precomputer.wait_for_completion(timeout=5.0)
            
            # Extract concepts and check each
            concepts = self.reasoner.extract_concepts(concept_or_text)
            
            for uri, confidence in concepts:
                if self.danger_precomputer.is_dangerous_uri(uri):
                    explanation = self.danger_precomputer.get_danger_explanation(uri)
                    path = [uri_to_label(uri)]
                    if explanation:
                        path.append(explanation)
                    
                    self._record_success()
                    self._measure_latency("is_dangerous", start_time)
                    return (True, path)
            
            self._record_success()
            self._measure_latency("is_dangerous", start_time)
            return (False, None)
            
        except (RateLimitExceeded, CircuitBreakerOpen):
            raise
        except Exception as e:
            self._record_failure()
            logger.error(f"Error in is_dangerous: {e}")
            raise
    
    def verify_response(self, text: str) -> tuple[bool, str]:
        """
        Verify an LLM response for safety and consistency.
        
        This is the primary method for using the engine as a firewall.
        It performs:
        1. Concept extraction
        2. Danger detection
        3. Consistency checking
        
        Args:
            text: The LLM response text to verify
            
        Returns:
            Tuple of (is_safe: bool, reason: str)
            - is_safe: True if no issues found
            - reason: Explanation of any issues, or "OK" if safe
            
        Raises:
            RateLimitExceeded: If rate limit is exceeded
            CircuitBreakerOpen: If circuit breaker is open
        """
        start_time = time.perf_counter()
        
        try:
            self._check_rate_limit()
            self._check_circuit_breaker()
            
            issues: list[str] = []
            
            # Wait for danger set if not ready
            if not self.danger_precomputer.danger_set_ready:
                self.danger_precomputer.wait_for_completion(timeout=5.0)
            
            # Check for dangerous content
            dangers = self.reasoner.check_danger_for_text(
                text, self.danger_precomputer
            )
            
            for danger in dangers:
                issues.append(
                    f"DANGER: {danger['explanation']}"
                )
            
            # Check for contradictions
            contradictions = self.reasoner.check_basic_consistency(text)
            
            for contradiction in contradictions:
                issues.append(
                    f"CONTRADICTION: {contradiction['explanation']}"
                )
            
            self._record_success()
            self._measure_latency("verify_response", start_time)
            
            if issues:
                return (False, " | ".join(issues))
            
            return (True, "OK")
            
        except RateLimitExceeded:
            return (False, "rate limit exceeded")
        except CircuitBreakerOpen:
            return (False, "circuit breaker open - service temporarily unavailable")
        except Exception as e:
            self._record_failure()
            logger.error(f"Error in verify_response: {e}")
            return (False, f"internal error: {str(e)}")
    
    def add_fact(self, head: str, rel: str, tail: str, weight: float = 1.0) -> bool:
        """
        Add a new fact to the knowledge base.
        
        This updates both the persistent database and the in-memory graph.
        If the fact affects danger propagation, the danger set is recomputed.
        
        Args:
            head: Head concept
            rel: Relation type
            tail: Tail concept
            weight: Confidence weight
            
        Returns:
            True if fact was added, False if it already existed
        """
        # Add to database
        added = self.data_loader.add_fact(head, rel, tail, weight)
        
        if added:
            # Update in-memory graph
            self.graph_engine.add_fact(head, rel, tail, weight)
            
            # Recompute danger set if relevant
            danger_relations = {"Causes", "HasProperty", "UsedFor", "CapableOf", "IsA"}
            if rel in danger_relations:
                logger.info("Recomputing danger set after adding fact...")
                self.danger_precomputer.start_computation()
        
        return added
    
    def get_stats(self) -> dict:
        """Get statistics about the engine."""
        return {
            "graph_nodes": self.graph_engine.node_count,
            "graph_edges": self.graph_engine.edge_count,
            "danger_set_ready": self.danger_precomputer.danger_set_ready,
            "danger_set_size": len(self.danger_precomputer.danger_set),
            "cached_embeddings": len(self.embedder.get_cached_uris()),
            "embedder_fallback_mode": self.embedder.is_fallback_mode,
            "circuit_state": self._circuit_state.value,
            "consecutive_failures": self._consecutive_failures,
        }
    
    def wait_for_ready(self, timeout: float = 30.0) -> bool:
        """
        Wait for the engine to be fully ready.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if ready, False if timeout
        """
        return self.danger_precomputer.wait_for_completion(timeout=timeout)
    
    @property
    def circuit_state(self) -> CircuitState:
        """Get current circuit breaker state."""
        return self._circuit_state
    
    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker to CLOSED state."""
        with self._circuit_lock:
            self._circuit_state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._circuit_timeout = 60.0
            logger.info("Circuit breaker manually reset to CLOSED")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Commonsense Reasoning Firewall - Validate LLM responses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --query "knife is sharp"
  %(prog)s --verify "Use a knife to cut vegetables"
  %(prog)s  # Interactive mode
        """
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="Query commonsense knowledge about a concept or text"
    )
    
    parser.add_argument(
        "--verify",
        type=str,
        help="Verify a text for safety and consistency"
    )
    
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/conceptnet.db",
        help="Path to ConceptNet SQLite database"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to sentence-transformers model"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize engine
    print("Initializing commonsense reasoning engine...")
    engine = FastCommonsenseEngine(
        db_path=args.db_path,
        model_path=args.model_path
    )
    
    # Wait for danger set computation
    print("Waiting for danger set computation...")
    engine.wait_for_ready(timeout=30.0)
    
    stats = engine.get_stats()
    print(f"Engine ready: {stats['graph_nodes']} nodes, {stats['graph_edges']} edges, "
          f"{stats['danger_set_size']} dangerous concepts")
    
    if args.query:
        # Query mode
        print(f"\nQuerying: {args.query}")
        print("-" * 50)
        
        inferences = engine.query_commonsense(args.query)
        
        if not inferences:
            print("No inferences found.")
        else:
            # Convert from cached tuple format back to dict
            for i, inf_tuple in enumerate(inferences, 1):
                inf = dict(inf_tuple)
                print(f"\n{i}. Score: {inf.get('score', 0):.3f}")
                print(f"   Explanation: {inf.get('explanation', 'N/A')}")
    
    elif args.verify:
        # Verify mode
        print(f"\nVerifying: {args.verify}")
        print("-" * 50)
        
        is_safe, reason = engine.verify_response(args.verify)
        
        if is_safe:
            print("SAFE: No issues detected")
        else:
            print(f"UNSAFE: {reason}")
    
    else:
        # Interactive mode
        print("\nInteractive mode. Commands:")
        print("  query <text>  - Query commonsense knowledge")
        print("  verify <text> - Verify text for safety")
        print("  danger <text> - Check if text is dangerous")
        print("  stats         - Show engine statistics")
        print("  quit          - Exit")
        print()
        
        while True:
            try:
                user_input = input("> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == "quit":
                    break
                
                if user_input.lower() == "stats":
                    stats = engine.get_stats()
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
                    continue
                
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print("Usage: <command> <text>")
                    continue
                
                command, text = parts
                command = command.lower()
                
                if command == "query":
                    inferences = engine.query_commonsense(text)
                    if not inferences:
                        print("No inferences found.")
                    else:
                        for i, inf_tuple in enumerate(inferences[:5], 1):
                            inf = dict(inf_tuple)
                            print(f"{i}. [{inf.get('score', 0):.2f}] {inf.get('explanation', 'N/A')}")
                
                elif command == "verify":
                    is_safe, reason = engine.verify_response(text)
                    if is_safe:
                        print("SAFE")
                    else:
                        print(f"UNSAFE: {reason}")
                
                elif command == "danger":
                    is_dangerous, explanation = engine.is_dangerous(text)
                    if is_dangerous:
                        print(f"DANGEROUS: {explanation}")
                    else:
                        print("NOT DANGEROUS")
                
                else:
                    print(f"Unknown command: {command}")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()
