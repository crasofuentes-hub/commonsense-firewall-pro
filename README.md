# CommonSense Firewall Enterprise (CS Firewall Enterprise)

Firewall semántico **determinista** y **offline-first** para analizar texto y detectar riesgos usando un grafo local tipo ConceptNet (SQLite).  
Diseñado para operar sin depender de APIs externas (p. ej. cuando ConceptNet público está caído) y para ofrecer **auditabilidad** mediante trazas reproducibles.

---

## ¿Qué problema resuelve?

Los sistemas basados en IA o en reglas suelen fallar por:
- **Alucinaciones** o inferencias sin soporte (“bullshit”).
- Falta de **auditoría** (no puedes justificar por qué se bloqueó o permitió).
- Dependencia de **servicios externos** (rate limits, caídas HTTP 502, latencia).

CS Firewall Enterprise aporta:
- Base de conocimiento **local** (SQLite) con millones de aristas.
- Evaluación de riesgo reproducible (misma entrada → misma salida).
- Reglas “hard” + evidencia del grafo para explicación.

---

## Garantías (estado actual)

| Propiedad | Estado |
|---|---|
| Offline-first (DB local) | ✅ |
| Determinismo operacional (misma entrada → misma salida) | ✅ (según CLI y pipeline actual) |
| Explicabilidad (hallazgos con evidencia) | ✅ |
| Ingest local desde ConceptNet `assertions.csv.gz` (TSV) | ✅ |
| Indexado y deduplicación (`UNIQUE (start_uri, rel, end_uri)`) | ✅ |
| CI / Formal verification | ⏳ (planificado) |

---

## Arquitectura (alto nivel)

**Entrada** (texto + entidades opcionales)  
→ **Normalización / extracción de conceptos**  
→ **Motor de reglas** (hard rules)  
→ **Motor de grafo (SQLite)** (búsqueda de relaciones / caminos)  
→ **Score de riesgo**  
→ **Decisión**: `ALLOW` / `BLOCK`  
→ **Salida JSON** con `findings` y explicación

Componentes principales:
- `config`: rutas y settings del sistema (DB local).
- `graph`: acceso a SQLite + consultas.
- `rules`: reglas deterministas de seguridad (p. ej. fuego/acelerantes/indoor).
- `cli`: `doctor`, `analyze`, `ingest`.

---

## Requisitos

- Windows / Linux / macOS
- Python 3.11+ (recomendado: **venv**)
- SQLite (incluido vía librería estándar de Python)
- Disco: la DB puede ser grande (cientos de MB o más)

---

## Instalación (Windows PowerShell)

```powershell
Set-Location C:\Users\servi\cs_firewall_enterprise

# 1) Crear venv
python -m venv .venv

# 2) Activar venv
.\.venv\Scripts\Activate.ps1

# 3) Instalar el paquete (editable)
python -m pip install -U pip
python -m pip install -e .
