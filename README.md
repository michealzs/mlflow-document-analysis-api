# Document Analysis API

A Flask API for analyzing text and PDF documents, with file-based MLFlow
experiment tracking. Documents are uploaded, stored in SQLite, chunked by word
count (without splitting sentences), and analyzed for a summary, keywords, or
sentiment. Each analysis is tracked as an MLFlow run and persisted to disk so it
can be retrieved and compared later. Results are cached for 5 minutes.

This is a standalone, runnable version of the API. All storage (SQLite database,
uploads, results, logs, and MLFlow runs) is created **relative to the `app/`
directory**, so it runs without root or Docker.

## Requirements

- Python 3.11 recommended (3.12 also works).
- Dependencies pinned in [`app/requirements.txt`](app/requirements.txt):
  - `Flask==3.1.0`, `Werkzeug==3.1.3`, `mlflow==2.21.3`, `PyPDF2==3.0.1`, `requests==2.32.3`
- Note: `mlflow` is a moderately heavy install (many transitive dependencies);
  the first `pip install` can take a few minutes.

## Setup

```bash
cd app
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the server

```bash
cd app
source .venv/bin/activate
python main.py
```

The server listens on `http://0.0.0.0:5000`. On startup it creates (under `app/`):

- `documents.db` — SQLite metadata store (with an index on `created_at`)
- `uploads/` — saved uploaded files
- `results/` — one JSON file per analysis, named `<analysis_id>.json`
- `logs/api.log` — one JSON line per request (`timestamp`, `method`, `path`, `status_code`)
- `mlruns/` — MLFlow file-based tracking, experiment `document-analysis`

A background thread runs hourly to purge expired cache entries and delete log
files older than 7 days.

## Quick start (curl)

The examples below assume the server is running. Sample files ship in
`app/sample_data/` (`sample.txt`, `sample.pdf`).

### 1. Health check

```bash
curl http://localhost:5000/health
# {"status":"ok"}
```

### 2. List available models

```bash
curl http://localhost:5000/models
```

```json
{
  "models": [
    {"name": "context-analyzer-v1", "context_window": 2048, "version": "1.0.0"},
    {"name": "long-doc-processor",  "context_window": 4096, "version": "2.1.0"},
    {"name": "summary-extractor",   "context_window": 8192, "version": "1.5.2"}
  ]
}
```

### 3. Upload a sample document

```bash
curl -X POST http://localhost:5000/documents \
  -F "file=@sample_data/sample.txt"
# {"document_id":"<uuid>","word_count":<N>}
```

Save the returned `document_id`; you will need it to analyze.

### 4. Analyze a document

```bash
DOC_ID=<paste document_id from step 3>
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d "{\"document_id\":\"$DOC_ID\",\"model_name\":\"context-analyzer-v1\",\"analysis_type\":\"keywords\"}"
```

```json
{
  "analysis_id": "<uuid>",
  "mlflow_run_id": "<run id>",
  "context_window": 2048,
  "num_chunks": 1,
  "results": [
    {"chunk_index": 0, "chunk_text": "...", "keywords": ["servers", "databases", ...]}
  ]
}
```

### 5. Retrieve results

```bash
ANALYSIS_ID=<paste analysis_id from step 4>
curl "http://localhost:5000/results/$ANALYSIS_ID"
# Add ?chunk_index=0 to get just one chunk
curl "http://localhost:5000/results/$ANALYSIS_ID?chunk_index=0"
```

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Returns `{"status": "ok"}`. |
| POST | `/documents` | Multipart upload (`file`). Accepts `.pdf` / `.txt`. Returns `201 {"document_id","word_count"}`. |
| GET | `/documents/<id>` | Returns `{"id","word_count","filename","created_at"}`; `404` if missing. |
| GET | `/documents` | Paginated list. `{"documents":[...],"total":N}`. |
| DELETE | `/documents/<id>` | Deletes file, DB row, and associated result files. `204` / `404`. |
| GET | `/models` | Returns the three built-in models. |
| POST | `/analyze` | Analyze one document. |
| GET | `/results/<analysis_id>` | Returns the persisted analysis; optional `?chunk_index=N`. |
| POST | `/batch/analyze` | Run a `jobs` array sequentially. |
| POST | `/compare` | Compare two persisted analyses. |

### `GET /documents` query parameters

- `limit` (default `10`), `offset` (default `0`) — must be non-negative, else `400`.
- `sort_by` — `created_at` (default) or `word_count`; anything else is `400`.
- `order` — `desc` (default) or `asc`; anything else is `400`.

### `POST /analyze` request body

```json
{
  "document_id": "<uuid>",
  "model_name": "context-analyzer-v1",
  "analysis_type": "summary",
  "context_window": 2048,
  "skip_cache": false
}
```

- `analysis_type` is one of `summary`, `keywords`, `sentiment` (default `summary`).
- `context_window` is optional; defaults to the model's max and must not exceed it (`400` if it does).
- Per-chunk output field: `summary` (first 100 chars), `keywords` (up to 10
  lower-cased words longer than 5 chars), or `sentiment` (always `"neutral"`).
- Status codes: `400` for missing/invalid `document_id`, invalid
  `analysis_type`, or oversized `context_window`; `404` for an unknown `model_name`.

### Chunking

Text is split into chunks **by word count, never mid-sentence**. Sentences are
accumulated into a chunk until adding the next sentence would exceed
`context_window` words; a single sentence longer than the window gets its own
chunk.

### `POST /batch/analyze`

```json
{ "jobs": [ { "document_id": "...", "model_name": "...", "analysis_type": "keywords" } ], "webhook_url": "https://..." }
```

Jobs run sequentially in input order. A failed job produces an `{"error": ...}`
object in its slot instead of crashing the batch. Returns
`{"batch_id","status":"completed","total_jobs","successful_jobs","failed_jobs","results"}`.
`webhook_url` is optional; if present the response is POSTed to it
(best-effort, with retry). Empty or missing `jobs` returns `400`.

### `POST /compare`

```json
{ "analysis_id_1": "<uuid>", "analysis_id_2": "<uuid>" }
```

Returns `{"common_keywords", "sentiment_agreement", "chunk_count_delta"}`:

- `common_keywords` — intersection of all keywords across both analyses.
- `sentiment_agreement` — fraction (0.0–1.0) of matching per-chunk sentiments
  over the overlapping chunk range; missing sentiments count as empty strings.
- `chunk_count_delta` — absolute difference in `num_chunks`.

Missing IDs return `400`; a missing result file returns `404`.

## Caching

Analysis results are cached in an in-memory LRU cache (capacity 100), keyed by
`document_id`, `model_name`, `analysis_type`, and `context_window`. Cached
entries expire after **5 minutes**. A cache hit returns the stored result with
`"cached": true` added. Send `"skip_cache": true` in the `/analyze` body to
bypass the cache (the fresh result is still written back to the cache).

## MLFlow tracking

Each analysis starts an MLFlow run under the `document-analysis` experiment
(`app/mlruns/`). Logged **parameters**: `document_id`, `model_name`,
`analysis_type`, `num_chunks`. Logged **metrics**: `avg_chunk_words`,
`max_chunk_words`. Browse runs with:

```bash
cd app
source .venv/bin/activate
mlflow ui --backend-store-uri "file://$(pwd)/mlruns"
```
