# My_personal_AI — Build a Vector Database in Python

A fully working educational **Vector Database** implemented in Python with a web UI. It demonstrates **HNSW**, **KD-Tree**, and **Brute Force** search algorithms, three distance metrics, a semantic-space visualization, and a **RAG pipeline** powered by a local LLM via Ollama.

> This project demonstrates how vector databases such as Pinecone, Weaviate, Chroma, and Milvus work under the hood. It is a learning prototype, not a production database.

---

## What This Project Does

| Feature | Description |
|---|---|
| **3 Search Algorithms** | HNSW, KD-Tree, and Brute Force search |
| **3 Distance Metrics** | Cosine similarity, Euclidean distance, Manhattan distance |
| **16D Demo Vectors** | 20 pre-loaded semantic vectors across CS, Math, Food, and Sports |
| **Semantic-Space Plot** | Visual view of demo vectors grouped by category |
| **Real Document Embedding** | Paste text and use Ollama's `nomic-embed-text` model |
| **RAG Pipeline** | Retrieve relevant document chunks and generate an answer with a local LLM |
| **REST API** | Insert, delete, search, benchmark, HNSW information, document search, and status endpoints |

## How It Works

```
Your Text
    |
    v
Ollama (nomic-embed-text)  <- converts text to an embedding vector
    |
    v
Vector Search              <- finds nearby vectors in semantic space
    |
    v
Retrieved Document Chunks  <- relevant context for the question
    |
    v
Ollama (llama3.2)          <- generates an answer
    |
    v
Answer
```

**HNSW (Hierarchical Navigable Small World)** organizes vectors into a layered neighbor graph. Searches navigate from upper layers toward closer neighbors at lower layers. The included implementation is intended for learning and small demonstrations.

---

## Prerequisites

You need **3 things** installed on your Windows laptop:

1. **Python 3.10 or newer**
2. **Git**
3. **Ollama** (optional for demo-vector search; required for document embeddings and RAG)

## Step-by-Step Setup (Windows)

### Step 1 — Install Python

1. Download Python from https://www.python.org/downloads/.
2. Run the installer and enable **Add Python to PATH**.
3. Open PowerShell and verify:

```powershell
python --version
pip --version
```

### Step 2 — Install Git

Download Git for Windows from https://git-scm.com/download/win and verify:

```powershell
git --version
```

### Step 3 — Install Ollama (Local AI Models)

1. Download Ollama from https://ollama.com.
2. Open PowerShell and pull the models:

```powershell
ollama pull nomic-embed-text
ollama pull llama3.2
```

3. Verify the models are available:

```powershell
ollama list
```

Ollama is only required for the Documents and Ask AI tabs. Demo vector search works without it. An 8 GB RAM system can run small local models, though generation may be slow.

### Step 4 — Clone the Repository

```powershell
git clone https://github.com/RohanKhandagale/My_Personal_AI.git
cd My_Personal_AI
```

### Step 5 — Install Python Dependencies

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, you can run `python -m pip install -r requirements.txt` without activating it.

### Step 6 — Run the Application

If Ollama is installed, start it if it is not already running:

```powershell
ollama serve
```

In another PowerShell window, start the Python web server:

```powershell
python app.py
```

Open your browser at http://localhost:8080.

The original 20 demo vectors load automatically. Demo search and benchmarking can be used without Ollama; document embedding and RAG require Ollama and the models listed above.

---

## Using the Application

### Semantic Search

- Search concepts such as `binary tree`, `sushi`, `basketball`, or `calculus`.
- Choose HNSW, KD-Tree, or Brute Force.
- Choose Cosine, Euclidean, or Manhattan distance.
- Change the number of nearest neighbors and view the ranked results.
- Compare algorithm timings with the benchmark control.
- Insert or delete demo vectors and inspect the vector visualization.

### Documents

1. Enter a document title and paste its text.
2. Click **Embed & Insert**.
3. Long documents are split into overlapping 250-word chunks.
4. Ollama creates an embedding for each chunk, which is stored for retrieval.

### Ask AI (RAG Pipeline)

1. Insert one or more documents first.
2. Enter a question and choose how many chunks to retrieve.
3. Click **Ask AI**.
4. The application embeds the question, retrieves similar document chunks, and sends them as context to `llama3.2`.

The document store is held in memory in this learning version; documents are cleared when the Python server restarts.

---

## REST API Reference

The server runs at `http://localhost:8080`.

### Demo Vector Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | K-nearest-neighbor search |
| `POST` | `/insert` | Insert a demo vector |
| `DELETE` | `/delete/:id` | Delete a demo vector |
| `GET` | `/items` | List demo vectors |
| `GET` | `/benchmark?v=...&k=5&metric=cosine` | Compare algorithm timings |
| `GET` | `/hnsw-info` | Graph nodes and layer statistics |
| `GET` | `/stats` | Database statistics |

### Document and RAG Endpoints

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/doc/insert` | `{"title":"...","text":"..."}` | Embed and store a document |
| `GET` | `/doc/list` | — | List stored document chunks |
| `DELETE` | `/doc/delete/:id` | — | Delete a document chunk |
| `POST` | `/doc/search` | `{"question":"...","k":3}` | Retrieve matching chunks |
| `POST` | `/doc/ask` | `{"question":"...","k":3}` | Retrieve context and generate an answer |
| `GET` | `/status` | — | Ollama status and model information |

### Example: Search via curl

```powershell
curl "http://localhost:8080/search?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1&k=3&metric=cosine&algo=hnsw"
```

### Example: Ask a Question via curl

```powershell
curl -X POST http://localhost:8080/doc/ask -H "Content-Type: application/json" -d "{\"question\":\"What is dynamic programming?\",\"k\":3}"
```

---

## Project Structure

```
My_personal_AI/
├── app.py              <- Python REST API, search logic, and Ollama RAG client
├── demo_vectors.json   <- Original 20 demo vectors
├── index.html          <- Web UI
├── requirements.txt    <- Python dependencies
├── tests.py            <- Automated tests
└── README.md           <- Setup and API documentation
```

## Algorithm Notes

- **Brute Force:** compares the query against every stored vector; a simple baseline.
- **KD-Tree:** a space-partitioning approach that is most useful for lower-dimensional data.
- **HNSW:** a layered neighbor graph designed for approximate nearest-neighbor search.
- **Cosine distance:** compares vector direction.
- **Euclidean distance:** measures straight-line distance.
- **Manhattan distance:** sums absolute coordinate differences.

KD-Tree search tends to degrade in high dimensions. HNSW is designed to keep searches efficient as the number and dimensionality of vectors grow, though this small educational implementation is not optimized for production-scale datasets.

## Common Issues

| Problem | Fix |
|---|---|
| `Ollama: OFFLINE` | Start Ollama with `ollama serve` |
| Embedding fails | Check Ollama is running and `nomic-embed-text` is installed |
| `python: command not found` | Reinstall Python and enable **Add Python to PATH** |
| Port 8080 already in use | Check with `netstat -ano | findstr 8080`, then stop the process if appropriate |
| LLM answer is slow | Local generation may be slow on CPU; try `ollama pull llama3.2:1b` and set `OLLAMA_GEN_MODEL=llama3.2:1b` before running `app.py` |

## Tests

Run the automated tests with:

```powershell
python -m unittest -v tests
```

## License
use this however you want.
