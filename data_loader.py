"""
DataLoader module for ConceptNet knowledge base management.

This module handles the connection to ConceptNet via SQLite and provides
high-level methods for querying and adding facts to the knowledge base.

EXPANDING THE KNOWLEDGE BASE:
=============================
To use the full ConceptNet dataset instead of the bootstrap mini-dataset:

1. Download ConceptNet assertions:
   wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz

2. Filter and import using pandas:
   ```python
   import pandas as pd
   import gzip
   
   # Read the compressed CSV
   with gzip.open('conceptnet-assertions-5.7.0.csv.gz', 'rt', encoding='utf-8') as f:
       df = pd.read_csv(f, sep='\\t', header=None, 
                        names=['uri', 'relation', 'head', 'tail', 'metadata'])
   
   # Parse weight from metadata JSON
   import json
   df['weight'] = df['metadata'].apply(lambda x: json.loads(x).get('weight', 1.0))
   
   # Filter for English, high-weight edges with relevant relations
   RELEVANT_RELATIONS = {
       '/r/IsA', '/r/UsedFor', '/r/Causes', '/r/HasProperty', '/r/CapableOf',
       '/r/PartOf', '/r/AtLocation', '/r/HasPrerequisite', '/r/Desires',
       '/r/Antonym', '/r/NotCapableOf', '/r/DistinctFrom'
   }
   
   filtered = df[
       (df['head'].str.startswith('/c/en/')) &
       (df['tail'].str.startswith('/c/en/')) &
       (df['relation'].isin(RELEVANT_RELATIONS)) &
       (df['weight'] > 1.0)
   ]
   
   # Import to SQLite using DataLoader.add_fact()
   loader = DataLoader('data/conceptnet.db')
   for _, row in filtered.iterrows():
       loader.add_fact(row['head'], row['relation'], row['tail'], row['weight'])
   ```

3. For ATOMIC integration, download ATOMIC 2020 and convert to (head, relation, tail) format:
   - ATOMIC events follow if-then patterns (cause->effect, intent->reaction)
   - Normalize relation names to match ConceptNet style
   - Use add_fact() to insert into the same database

Author: Commonsense Firewall Team
License: MIT
"""

import sqlite3
import logging
import os
from typing import Optional
from functools import lru_cache

logger = logging.getLogger(__name__)


# Bootstrap mini-dataset with 30 triples covering physical concepts and safety properties
# These triples are in ConceptNet format and provide basic commonsense knowledge
BOOTSTRAP_TRIPLES: list[dict] = [
    # Dangerous objects and their properties
    {"head": "knife", "rel": "IsA", "tail": "weapon", "weight": 2.0},
    {"head": "knife", "rel": "UsedFor", "tail": "cutting", "weight": 3.0},
    {"head": "knife", "rel": "HasProperty", "tail": "sharp", "weight": 2.5},
    {"head": "knife", "rel": "CapableOf", "tail": "cause_injury", "weight": 2.0},
    
    {"head": "gun", "rel": "IsA", "tail": "weapon", "weight": 3.0},
    {"head": "gun", "rel": "CapableOf", "tail": "kill", "weight": 3.0},
    {"head": "gun", "rel": "HasProperty", "tail": "dangerous", "weight": 3.0},
    {"head": "gun", "rel": "UsedFor", "tail": "shooting", "weight": 2.5},
    
    {"head": "poison", "rel": "Causes", "tail": "death", "weight": 3.0},
    {"head": "poison", "rel": "HasProperty", "tail": "dangerous", "weight": 3.0},
    {"head": "poison", "rel": "HasProperty", "tail": "toxic", "weight": 2.5},
    
    {"head": "fire", "rel": "Causes", "tail": "burn", "weight": 3.0},
    {"head": "fire", "rel": "HasProperty", "tail": "hot", "weight": 3.0},
    {"head": "fire", "rel": "CapableOf", "tail": "destroy", "weight": 2.5},
    {"head": "burn", "rel": "Causes", "tail": "injury", "weight": 2.5},
    
    {"head": "electricity", "rel": "CapableOf", "tail": "electrocute", "weight": 2.5},
    {"head": "electricity", "rel": "HasProperty", "tail": "dangerous", "weight": 2.0},
    {"head": "electrocute", "rel": "Causes", "tail": "death", "weight": 2.5},
    
    {"head": "explosion", "rel": "Causes", "tail": "death", "weight": 3.0},
    {"head": "explosion", "rel": "Causes", "tail": "injury", "weight": 3.0},
    {"head": "explosion", "rel": "HasProperty", "tail": "dangerous", "weight": 3.0},
    
    # Safe/neutral concepts
    {"head": "water", "rel": "HasProperty", "tail": "wet", "weight": 3.0},
    {"head": "water", "rel": "UsedFor", "tail": "drinking", "weight": 2.5},
    {"head": "water", "rel": "HasProperty", "tail": "liquid", "weight": 2.5},
    
    {"head": "medicine", "rel": "UsedFor", "tail": "healing", "weight": 3.0},
    {"head": "medicine", "rel": "HasProperty", "tail": "helpful", "weight": 2.0},
    
    {"head": "child", "rel": "HasProperty", "tail": "vulnerable", "weight": 2.5},
    {"head": "child", "rel": "Desires", "tail": "safety", "weight": 2.0},
    
    # Antonyms and contradictions for consistency checking
    {"head": "wet", "rel": "Antonym", "tail": "dry", "weight": 3.0},
    {"head": "hot", "rel": "Antonym", "tail": "cold", "weight": 3.0},
    {"head": "safe", "rel": "Antonym", "tail": "dangerous", "weight": 3.0},
    {"head": "legal", "rel": "Antonym", "tail": "illegal", "weight": 3.0},
    {"head": "harmless", "rel": "Antonym", "tail": "harmful", "weight": 3.0},
    
    # Legal/illegal concepts
    {"head": "weapon", "rel": "HasProperty", "tail": "dangerous", "weight": 2.5},
    {"head": "murder", "rel": "HasProperty", "tail": "illegal", "weight": 3.0},
    {"head": "murder", "rel": "Causes", "tail": "death", "weight": 3.0},
    {"head": "arson", "rel": "HasProperty", "tail": "illegal", "weight": 3.0},
    {"head": "arson", "rel": "Causes", "tail": "fire", "weight": 2.5},
    
    # Physical properties for contradiction detection
    {"head": "ice", "rel": "HasProperty", "tail": "cold", "weight": 3.0},
    {"head": "ice", "rel": "IsA", "tail": "solid", "weight": 2.5},
    
    # Capability chains
    {"head": "car", "rel": "CapableOf", "tail": "transport", "weight": 2.5},
    {"head": "car", "rel": "CapableOf", "tail": "cause_accident", "weight": 2.0},
    {"head": "accident", "rel": "Causes", "tail": "injury", "weight": 2.5},
    
    # Location and part-of relationships
    {"head": "blade", "rel": "PartOf", "tail": "knife", "weight": 2.5},
    {"head": "trigger", "rel": "PartOf", "tail": "gun", "weight": 2.5},
    
    # Prerequisites
    {"head": "burn", "rel": "HasPrerequisite", "tail": "fire", "weight": 2.5},
    {"head": "electrocute", "rel": "HasPrerequisite", "tail": "electricity", "weight": 2.5},
]


def normalize_to_uri(concept: str) -> str:
    """
    Normalize a concept string to ConceptNet URI format.
    
    Args:
        concept: A concept string (e.g., "knife", "cause_injury", "/c/en/knife")
        
    Returns:
        ConceptNet URI format string (e.g., "/c/en/knife")
    """
    if concept.startswith("/c/en/"):
        return concept
    cleaned = concept.lower().strip().replace(" ", "_").replace("-", "_")
    return f"/c/en/{cleaned}"


def uri_to_label(uri: str) -> str:
    """
    Convert a ConceptNet URI to a human-readable label.
    
    Args:
        uri: ConceptNet URI (e.g., "/c/en/knife")
        
    Returns:
        Human-readable label (e.g., "knife")
    """
    if uri.startswith("/c/en/"):
        return uri[6:].replace("_", " ")
    return uri


class DataLoader:
    """
    Manages the ConceptNet knowledge base stored in SQLite.
    
    This class provides methods to:
    - Connect to and initialize the SQLite database
    - Query edges for concepts
    - Add new facts to the knowledge base
    - Bootstrap the database with a mini-dataset if empty
    
    The database schema follows a simple edge-list format:
    - edges table: (id, head_uri, relation, tail_uri, weight)
    
    For production use with full ConceptNet:
    - Download conceptnet-assertions-5.7.0.csv.gz
    - Filter for English, weight > 1.0, relevant relations
    - Use add_fact() to populate the database
    
    For ATOMIC integration:
    - ATOMIC uses if-then event patterns
    - Normalize to (head, relation, tail) format
    - Relations like xIntent, xReact, oReact map to causal chains
    """
    
    def __init__(self, db_path: str = "data/conceptnet.db"):
        """
        Initialize the DataLoader with a path to the SQLite database.
        
        Args:
            db_path: Path to the SQLite database file. Will be created if it doesn't exist.
        """
        self.db_path = db_path
        self._ensure_directory()
        self._init_database()
        
    def _ensure_directory(self) -> None:
        """Ensure the directory for the database file exists."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            logger.info(f"Created directory: {db_dir}")
    
    def _init_database(self) -> None:
        """Initialize the database schema if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    head_uri TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    tail_uri TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    UNIQUE(head_uri, relation, tail_uri)
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_head_uri ON edges(head_uri)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_tail_uri ON edges(tail_uri)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_relation ON edges(relation)
            """)
            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
    
    def ensure_bootstrap_data(self) -> None:
        """
        Ensure the database contains at least the bootstrap mini-dataset.
        
        If the database is empty, this method inserts the BOOTSTRAP_TRIPLES
        to provide basic commonsense knowledge for testing and development.
        
        In production, you should replace this with a full ConceptNet import:
        1. Download conceptnet-assertions-5.7.0.csv.gz
        2. Filter for relevant relations and weight > 1.0
        3. Use add_fact() to populate the database
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM edges")
            count = cursor.fetchone()[0]
            
            if count == 0:
                logger.info("Database is empty, inserting bootstrap data...")
                for triple in BOOTSTRAP_TRIPLES:
                    self.add_fact(
                        triple["head"],
                        triple["rel"],
                        triple["tail"],
                        triple["weight"]
                    )
                logger.info(f"Inserted {len(BOOTSTRAP_TRIPLES)} bootstrap triples")
            else:
                logger.info(f"Database already contains {count} edges")
    
    def add_fact(self, head: str, rel: str, tail: str, weight: float = 1.0) -> bool:
        """
        Add a new fact (edge) to the knowledge base.
        
        Args:
            head: The head concept (will be normalized to URI format)
            rel: The relation type (e.g., "IsA", "Causes", "/r/IsA")
            tail: The tail concept (will be normalized to URI format)
            weight: The confidence weight of this fact (default 1.0)
            
        Returns:
            True if the fact was added, False if it already exists
            
        Example:
            >>> loader.add_fact("knife", "IsA", "weapon", 2.0)
            True
            >>> loader.add_fact("knife", "IsA", "weapon", 2.0)  # duplicate
            False
        """
        head_uri = normalize_to_uri(head)
        tail_uri = normalize_to_uri(tail)
        
        # Normalize relation (remove /r/ prefix if present)
        if rel.startswith("/r/"):
            rel = rel[3:]
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR IGNORE INTO edges (head_uri, relation, tail_uri, weight) VALUES (?, ?, ?, ?)",
                    (head_uri, rel, tail_uri, weight)
                )
                conn.commit()
                if cursor.rowcount > 0:
                    logger.debug(f"Added fact: {head_uri} --[{rel}]--> {tail_uri}")
                    return True
                return False
        except sqlite3.Error as e:
            logger.error(f"Error adding fact: {e}")
            return False
    
    def get_edges_for_concept(self, uri: str) -> list[tuple[str, str, str, float]]:
        """
        Get all edges where the given concept is the head.
        
        Args:
            uri: The concept URI (will be normalized if not already in URI format)
            
        Returns:
            List of tuples (head_uri, relation, tail_uri, weight)
            
        Example:
            >>> loader.get_edges_for_concept("knife")
            [('/c/en/knife', 'IsA', '/c/en/weapon', 2.0),
             ('/c/en/knife', 'UsedFor', '/c/en/cutting', 3.0)]
        """
        uri = normalize_to_uri(uri)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT head_uri, relation, tail_uri, weight FROM edges WHERE head_uri = ?",
                (uri,)
            )
            return cursor.fetchall()
    
    def get_incoming_edges(self, uri: str) -> list[tuple[str, str, str, float]]:
        """
        Get all edges where the given concept is the tail.
        
        Args:
            uri: The concept URI (will be normalized if not already in URI format)
            
        Returns:
            List of tuples (head_uri, relation, tail_uri, weight)
        """
        uri = normalize_to_uri(uri)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT head_uri, relation, tail_uri, weight FROM edges WHERE tail_uri = ?",
                (uri,)
            )
            return cursor.fetchall()
    
    def get_all_edges(self) -> list[tuple[str, str, str, float]]:
        """
        Get all edges in the knowledge base.
        
        Returns:
            List of tuples (head_uri, relation, tail_uri, weight)
            
        Note: For large databases, consider using pagination or streaming.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT head_uri, relation, tail_uri, weight FROM edges")
            return cursor.fetchall()
    
    def get_all_concepts(self) -> set[str]:
        """
        Get all unique concept URIs in the knowledge base.
        
        Returns:
            Set of all concept URIs (both heads and tails)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT head_uri FROM edges UNION SELECT DISTINCT tail_uri FROM edges")
            return {row[0] for row in cursor.fetchall()}
    
    def get_edges_by_relation(self, relation: str) -> list[tuple[str, str, str, float]]:
        """
        Get all edges with a specific relation type.
        
        Args:
            relation: The relation type (e.g., "IsA", "Causes")
            
        Returns:
            List of tuples (head_uri, relation, tail_uri, weight)
        """
        # Normalize relation
        if relation.startswith("/r/"):
            relation = relation[3:]
            
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT head_uri, relation, tail_uri, weight FROM edges WHERE relation = ?",
                (relation,)
            )
            return cursor.fetchall()
    
    def concept_exists(self, uri: str) -> bool:
        """
        Check if a concept exists in the knowledge base.
        
        Args:
            uri: The concept URI to check
            
        Returns:
            True if the concept exists as head or tail of any edge
        """
        uri = normalize_to_uri(uri)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM edges WHERE head_uri = ? OR tail_uri = ? LIMIT 1",
                (uri, uri)
            )
            return cursor.fetchone() is not None
    
    def get_edge_count(self) -> int:
        """Get the total number of edges in the knowledge base."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM edges")
            return cursor.fetchone()[0]
    
    def clear_database(self) -> None:
        """
        Clear all edges from the database.
        
        WARNING: This is destructive and cannot be undone.
        Use only for testing or re-initialization.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM edges")
            conn.commit()
            logger.warning("Database cleared")


# ATOMIC Integration Notes:
# =========================
# ATOMIC (https://allenai.org/data/atomic-2020) provides if-then knowledge:
# - xIntent: PersonX's intent
# - xReact: PersonX's reaction
# - oReact: Others' reaction
# - xNeed: What PersonX needs before
# - xWant: What PersonX wants after
# - xEffect: Effect on PersonX
# - oEffect: Effect on others
#
# To integrate ATOMIC:
# 1. Download atomic2020_data-feb2021.zip
# 2. Parse the TSV files
# 3. Map ATOMIC relations to ConceptNet-style relations:
#    - xIntent -> HasPrerequisite (intent before action)
#    - xReact/oReact -> Causes (action causes reaction)
#    - xNeed -> HasPrerequisite
#    - xWant -> Desires
#    - xEffect/oEffect -> Causes
# 4. Use add_fact() to insert normalized triples
#
# Example ATOMIC entry:
#   "PersonX burns PersonY's house" -> xEffect -> "PersonX goes to jail"
# Normalized:
#   add_fact("burn_house", "Causes", "go_to_jail", 2.0)
