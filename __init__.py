"""
Commonsense Reasoning Firewall

A 100% offline commonsense reasoning engine that acts as a logical firewall
for LLM responses, detecting hallucinations, contradictions, and safety violations.

Usage:
    from commonsense_firewall import FastCommonsenseEngine
    
    engine = FastCommonsenseEngine()
    is_safe, reason = engine.verify_response("LLM response text")
"""

from engine import FastCommonsenseEngine
from data_loader import DataLoader
from graph_engine import GraphEngine
from danger_precomputer import DangerPrecomputer
from semantic_embedder import SemanticEmbedder
from reasoner import Reasoner

__version__ = "1.0.0"
__all__ = [
    "FastCommonsenseEngine",
    "DataLoader",
    "GraphEngine",
    "DangerPrecomputer",
    "SemanticEmbedder",
    "Reasoner",
]
