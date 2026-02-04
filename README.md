# CommonSense Firewall Enterprise (CSFW Enterprise)

**Offline-first, deterministic semantic safety firewall** that inspects text (and/or declared entities) before it reaches an LLM, an agent runtime, or an OS/action layer.  
It uses a **local ConceptNet-style knowledge graph** stored in **SQLite** to provide **explainable, reproducible risk decisions** without relying on external APIs (which can rate-limit, drift, or go down).

> Status: **Working end-to-end** (local DB ingestion + CLI analyze + doctor).  
> ConceptNet public API outages (HTTP 502) are handled by **offline ingestion from `assertions.tsv.gz`** (often shipped as `assertions.csv.gz` but actually TSV).

---

## Why this exists (Motivation)

Modern agent systems can fail in predictable ways:

- **Non-deterministic recovery**: repeated runs yield different “fixes”.
- **Unverifiable reasoning**: there’s no audit trail that ties the decision to evidence.
- **External dependency fragility**: reliance on live knowledge APIs leads to downtime (e.g., ConceptNet 502), rate limits, or drift.
- **Unsafe instruction pathways**: an agent can interpret a request as operational guidance and perform harmful steps.

CSFW Enterprise aims to be the **semantic safety boundary** that:
1. Runs **offline** against a local KB,
2. Produces a **deterministic verdict** (`ALLOW`/`BLOCK`) + evidence,
3. Is designed to integrate into a **fixpoint + hashed trace** pipeline for convergence and anti-loop guarantees (planned integration with `reflexive-dsha`).

---

## Key properties (Guarantees)

| Capability | Current |
|---|---|
| Offline-first local knowledge base (SQLite) | ✅ |
| Deterministic operational behavior (same input → same output) | ✅ (current CLI pipeline) |
| Explainable decisions with evidence | ✅ |
| Hard-rule safety blocks for critical intents | ✅ |
| High-volume offline ingestion from ConceptNet assertions dump | ✅ |
| De-duplication at storage layer via UNIQUE triplet index | ✅ |
| Formal proofs (Lean) for fixpoint termination/idempotence/trace soundness | ⏳ Planned / not yet ported here |

---

## System overview (Architecture)

High-level flow:

**Text + Entities**  
→ (optional) entity normalization  
→ **Hard Rules Engine** (fast, deterministic blocks)  
→ **Graph Evidence Engine** (SQLite queries over `edges`)  
→ Risk scoring  
→ **Decision** (`ALLOW`/`BLOCK`)  
→ **JSON report**: entities, findings, evidence, explanation

Core components:
- **Config / settings**: DB path, defaults.
- **Graph repository**: SQLite-backed edge queries.
- **Rules**: deterministic, “hard-stop” constraints (e.g., accelerant + indoors + ignition).
- **CLI**:
  - `doctor`: DB health check
  - `analyze`: analyze text with declared entities
  - `ingest`: import ConceptNet assertions dump into local DB

---

## Repository layout (typical)

