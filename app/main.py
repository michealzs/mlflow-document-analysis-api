import hashlib
import json
import os
import re
import sqlite3
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta
from collections import OrderedDict

import mlflow
from flask import Flask, jsonify, request

app = Flask(__name__)

# Resolve all storage paths relative to this file so the app runs without
# root privileges or Docker. In the original container these were under /app.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
DB_PATH = os.path.join(BASE_DIR, "documents.db")
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_PATH = os.path.join(LOGS_DIR, "api.log")

# MLFlow setup
mlflow.set_tracking_uri("file://" + MLRUNS_DIR)
mlflow.set_experiment("document-analysis")

MODELS = [
    {"name": "context-analyzer-v1", "context_window": 2048, "version": "1.0.0"},
    {"name": "long-doc-processor", "context_window": 4096, "version": "2.1.0"},
    {"name": "summary-extractor", "context_window": 8192, "version": "1.5.2"},
]

# LRU Cache implementation with TTL
class LRUCache:
    def __init__(self, capacity=100):
        self.cache = OrderedDict()
        self.capacity = capacity
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key not in self.cache:
                return None
            entry = self.cache[key]
            if time.time() - entry["timestamp"] > 300:
                del self.cache[key]
                return None
            self.cache.move_to_end(key)
            return entry["data"]

    def put(self, key, value):
        with self._lock:
            self.cache[key] = {"data": value, "timestamp": time.time()}
            self.cache.move_to_end(key)
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

_analysis_cache = LRUCache(capacity=100)

def _get_cache_key(document_id, model_name, analysis_type, context_window):
    key_data = f"{document_id}:{model_name}:{analysis_type}:{context_window}"
    return hashlib.md5(key_data.encode()).hexdigest()


def _log_request(method, path, status_code):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "method": method,
        "path": path,
        "status_code": status_code
    }
    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Failed to write to log: {e}", file=sys.stderr)


# Background cleanup thread
def _cleanup_expired_cache():
    """Remove expired cache entries."""
    current_time = time.time()
    expired_count = 0
    with _analysis_cache._lock:
        expired_keys = [
            key for key, entry in _analysis_cache.cache.items()
            if current_time - entry["timestamp"] > 300
        ]
        for key in expired_keys:
            del _analysis_cache.cache[key]
            expired_count += 1
    if expired_count > 0:
        print(f"Cleaned up {expired_count} expired cache entries", file=sys.stderr)


def _cleanup_old_log_files():
    """Remove log files older than 7 days from the logs directory."""
    cutoff_time = time.time() - (7 * 24 * 60 * 60)  # 7 days in seconds
    removed_count = 0
    try:
        if os.path.exists(LOGS_DIR):
            for filename in os.listdir(LOGS_DIR):
                filepath = os.path.join(LOGS_DIR, filename)
                if os.path.isfile(filepath):
                    try:
                        stat = os.stat(filepath)
                        if stat.st_mtime < cutoff_time:
                            os.remove(filepath)
                            removed_count += 1
                    except Exception:
                        pass
    except Exception as e:
        print(f"Error cleaning up log files: {e}", file=sys.stderr)
    if removed_count > 0:
        print(f"Cleaned up {removed_count} old log files", file=sys.stderr)


def _cleanup_worker():
    """Background thread that runs cleanup every hour."""
    while True:
        try:
            time.sleep(3600)  # Sleep for 1 hour
            _cleanup_expired_cache()
            _cleanup_old_log_files()
        except Exception as e:
            print(f"Cleanup worker error: {e}", file=sys.stderr)


# Start the background cleanup thread
cleanup_thread = threading.Thread(target=_cleanup_worker, daemon=True)
cleanup_thread.start()


@app.after_request
def after_request(response):
    _log_request(request.method, request.path, response.status_code)
    return response


def _init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                word_count INTEGER NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                analysis_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                analysis_type TEXT NOT NULL,
                context_window INTEGER NOT NULL,
                num_chunks INTEGER NOT NULL,
                mlflow_run_id TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at)"
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database initialization error: {e}", file=sys.stderr)
        raise


_init_db()


def _get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _sentences(text):
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


def chunk_text(text, context_window):
    sentences = _sentences(text)
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        if current_word_count > 0 and current_word_count + sentence_word_count > context_window:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = sentence_word_count
        else:
            current_chunk.append(sentence)
            current_word_count += sentence_word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def extract_text_from_pdf(filepath):
    from PyPDF2 import PdfReader
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    if not text.strip():
        raise ValueError("Could not extract text from PDF")
    return text


def extract_text_from_txt(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def _deliver_webhook(webhook_url, payload):
    """Deliver webhook with exponential backoff retry."""
    import requests
    delays = [1, 2, 4, 8]  # Exponential backoff
    for attempt, delay in enumerate(delays):
        try:
            resp = requests.post(webhook_url, json=payload, timeout=10)
            if resp.status_code < 500:
                return True
        except Exception:
            pass
        if attempt < len(delays) - 1:
            time.sleep(delay)
    return False


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/documents", methods=["POST"])
def upload_document():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if ".." in file.filename:
        return jsonify({"error": "Invalid filename"}), 400

    safe_filename = os.path.basename(file.filename)

    if not (safe_filename.endswith(".pdf") or safe_filename.endswith(".txt")):
        return jsonify({"error": "Unsupported file type"}), 400

    doc_id = str(uuid.uuid4())
    filepath = os.path.join(UPLOAD_DIR, f"{doc_id}_{safe_filename}")

    conn = _get_db()
    try:
        file.save(filepath)

        try:
            if safe_filename.endswith(".pdf"):
                content = extract_text_from_pdf(filepath)
            elif safe_filename.endswith(".txt"):
                content = extract_text_from_txt(filepath)
        except Exception as e:
            os.remove(filepath)
            return jsonify({"error": str(e)}), 400

        word_count = len(content.split())
        created_at = datetime.utcnow().isoformat()

        conn.execute(
            "INSERT INTO documents (id, filename, filepath, word_count, content, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (doc_id, safe_filename, filepath, word_count, content, created_at),
        )
        conn.commit()

        return jsonify({"document_id": doc_id, "word_count": word_count}), 201
    except Exception as e:
        conn.rollback()
        if os.path.exists(filepath):
            os.remove(filepath)
        print(f"Upload error: {e}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


@app.route("/documents/<doc_id>", methods=["GET"])
def get_document(doc_id):
    conn = _get_db()
    row = conn.execute(
        "SELECT id, word_count, filename, created_at FROM documents WHERE id = ?", (doc_id,)
    ).fetchone()
    conn.close()

    if row is None:
        return jsonify({"error": "Document not found"}), 404

    return jsonify({
        "id": row["id"],
        "word_count": row["word_count"],
        "filename": row["filename"],
        "created_at": row["created_at"],
    })


@app.route("/documents/<doc_id>", methods=["DELETE"])
def delete_document(doc_id):
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT filepath FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()

        if row is None:
            conn.close()
            return jsonify({"error": "Document not found"}), 404

        filepath = row["filepath"]

        if os.path.exists(filepath):
            os.remove(filepath)

        analyses = conn.execute(
            "SELECT analysis_id FROM analyses WHERE document_id = ?", (doc_id,)
        ).fetchall()
        for a in analyses:
            result_path = os.path.join(RESULTS_DIR, f"{a['analysis_id']}.json")
            if os.path.exists(result_path):
                os.remove(result_path)

        conn.execute("DELETE FROM analyses WHERE document_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()

        return "", 204
    except Exception as e:
        conn.rollback()
        print(f"Delete error: {e}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


@app.route("/documents", methods=["GET"])
def list_documents():
    try:
        limit = int(request.args.get("limit", 10))
        offset = int(request.args.get("offset", 0))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid pagination parameters"}), 400

    if limit < 0 or offset < 0:
        return jsonify({"error": "limit and offset must be non-negative"}), 400

    sort_by = request.args.get("sort_by", "created_at")
    order = request.args.get("order", "desc")
    if sort_by not in ("word_count", "created_at"):
        return jsonify({"error": "Invalid sort_by"}), 400
    if order not in ("asc", "desc"):
        return jsonify({"error": "Invalid order"}), 400

    direction = "DESC" if order == "desc" else "ASC"

    conn = _get_db()
    total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    rows = conn.execute(
        f"SELECT id, word_count, filename, created_at FROM documents ORDER BY {sort_by} {direction} LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    conn.close()

    return jsonify({
        "documents": [
            {
                "id": r["id"],
                "word_count": r["word_count"],
                "filename": r["filename"],
                "created_at": r["created_at"],
            }
            for r in rows
        ],
        "total": total,
    })


@app.route("/models", methods=["GET"])
def list_models():
    return jsonify({"models": MODELS})


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    doc_id = data.get("document_id")
    model_name = data.get("model_name")
    context_window = data.get("context_window")
    analysis_type = data.get("analysis_type", "summary")
    skip_cache = data.get("skip_cache", False)

    if not doc_id:
        return jsonify({"error": "document_id is required"}), 400

    conn = _get_db()
    doc = conn.execute(
        "SELECT id, content FROM documents WHERE id = ?", (doc_id,)
    ).fetchone()
    conn.close()

    if doc is None:
        return jsonify({"error": "Document not found"}), 400

    if not model_name:
        return jsonify({"error": "model_name is required"}), 400

    model = next((m for m in MODELS if m["name"] == model_name), None)
    if not model:
        return jsonify({"error": "Model not found"}), 404

    if context_window is None:
        context_window = model["context_window"]
    if context_window > model["context_window"]:
        return jsonify({"error": "context_window exceeds model maximum"}), 400

    if analysis_type not in ("summary", "keywords", "sentiment"):
        return jsonify({"error": "Unsupported analysis_type"}), 400

    cache_key = _get_cache_key(doc_id, model_name, analysis_type, context_window)
    if not skip_cache:
        cached = _analysis_cache.get(cache_key)
        if cached:
            cached["cached"] = True
            return jsonify(cached), 200

    chunks = chunk_text(doc["content"], context_window)

    run = mlflow.start_run()
    mlflow.log_param("document_id", doc_id)
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("analysis_type", analysis_type)
    mlflow.log_param("num_chunks", len(chunks))

    results = []
    chunk_word_counts = []
    for idx, chunk in enumerate(chunks):
        chunk_word_counts.append(len(chunk.split()))
        if analysis_type == "summary":
            result = {"chunk_index": idx, "chunk_text": chunk, "summary": chunk[:100]}
        elif analysis_type == "keywords":
            words = chunk.split()
            result = {
                "chunk_index": idx,
                "chunk_text": chunk,
                "keywords": list(set(w.lower() for w in words if len(w) > 5))[:10],
            }
        elif analysis_type == "sentiment":
            result = {"chunk_index": idx, "chunk_text": chunk, "sentiment": "neutral"}
        results.append(result)

    if chunk_word_counts:
        mlflow.log_metric("avg_chunk_words", sum(chunk_word_counts) / len(chunk_word_counts))
        mlflow.log_metric("max_chunk_words", max(chunk_word_counts))
    mlflow.end_run()

    analysis_id = str(uuid.uuid4())

    output = {
        "analysis_id": analysis_id,
        "document_id": doc_id,
        "model_name": model_name,
        "context_window": context_window,
        "analysis_type": analysis_type,
        "num_chunks": len(chunks),
        "mlflow_run_id": run.info.run_id,
        "results": results,
    }

    with open(os.path.join(RESULTS_DIR, f"{analysis_id}.json"), "w") as f:
        json.dump(output, f)

    conn = _get_db()
    conn.execute(
        "INSERT INTO analyses (analysis_id, document_id, model_name, analysis_type, context_window, num_chunks, mlflow_run_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (analysis_id, doc_id, model_name, analysis_type, context_window, len(chunks), run.info.run_id),
    )
    conn.commit()
    conn.close()

    _analysis_cache.put(cache_key, output)

    return jsonify(output), 200


@app.route("/batch/analyze", methods=["POST"])
def batch_analyze():
    data = request.get_json(silent=True) or {}
    jobs = data.get("jobs")
    webhook_url = data.get("webhook_url")

    if not jobs or not isinstance(jobs, list):
        return jsonify({"error": "jobs array is required"}), 400

    batch_id = str(uuid.uuid4())
    results = []
    successful_jobs = 0
    failed_jobs = 0

    for job in jobs:
        try:
            doc_id = job.get("document_id")
            model_name = job.get("model_name")
            context_window = job.get("context_window")
            analysis_type = job.get("analysis_type", "summary")

            if not doc_id:
                results.append({"error": "Invalid or missing document_id"})
                failed_jobs += 1
                continue

            conn = _get_db()
            doc = conn.execute(
                "SELECT id, content FROM documents WHERE id = ?", (doc_id,)
            ).fetchone()
            conn.close()

            if doc is None:
                results.append({"error": "Invalid or missing document_id"})
                failed_jobs += 1
                continue

            model = next((m for m in MODELS if m["name"] == model_name), None)
            if not model:
                results.append({"error": "Model not found"})
                failed_jobs += 1
                continue

            if context_window is None:
                context_window = model["context_window"]
            if context_window > model["context_window"]:
                results.append({"error": "context_window exceeds model maximum"})
                failed_jobs += 1
                continue

            if analysis_type not in ("summary", "keywords", "sentiment"):
                results.append({"error": "Unsupported analysis_type"})
                failed_jobs += 1
                continue

            chunks = chunk_text(doc["content"], context_window)

            run = mlflow.start_run()
            mlflow.log_param("document_id", doc_id)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("analysis_type", analysis_type)
            mlflow.log_param("num_chunks", len(chunks))

            job_results = []
            chunk_word_counts = []
            for idx, chunk in enumerate(chunks):
                chunk_word_counts.append(len(chunk.split()))
                if analysis_type == "summary":
                    result = {"chunk_index": idx, "chunk_text": chunk, "summary": chunk[:100]}
                elif analysis_type == "keywords":
                    words = chunk.split()
                    result = {
                        "chunk_index": idx,
                        "chunk_text": chunk,
                        "keywords": list(set(w.lower() for w in words if len(w) > 5))[:10],
                    }
                elif analysis_type == "sentiment":
                    result = {"chunk_index": idx, "chunk_text": chunk, "sentiment": "neutral"}
                job_results.append(result)

            if chunk_word_counts:
                mlflow.log_metric("avg_chunk_words", sum(chunk_word_counts) / len(chunk_word_counts))
                mlflow.log_metric("max_chunk_words", max(chunk_word_counts))
            mlflow.end_run()

            analysis_id = str(uuid.uuid4())

            output = {
                "analysis_id": analysis_id,
                "document_id": doc_id,
                "model_name": model_name,
                "context_window": context_window,
                "analysis_type": analysis_type,
                "num_chunks": len(chunks),
                "mlflow_run_id": run.info.run_id,
                "results": job_results,
            }

            with open(os.path.join(RESULTS_DIR, f"{analysis_id}.json"), "w") as f:
                json.dump(output, f)

            conn = _get_db()
            conn.execute(
                "INSERT INTO analyses (analysis_id, document_id, model_name, analysis_type, context_window, num_chunks, mlflow_run_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (analysis_id, doc_id, model_name, analysis_type, context_window, len(chunks), run.info.run_id),
            )
            conn.commit()
            conn.close()

            results.append({
                "analysis_id": analysis_id,
                "mlflow_run_id": run.info.run_id,
            })
            successful_jobs += 1
        except Exception as e:
            results.append({"error": str(e)})
            failed_jobs += 1

    response_data = {
        "batch_id": batch_id,
        "status": "completed",
        "total_jobs": len(jobs),
        "successful_jobs": successful_jobs,
        "failed_jobs": failed_jobs,
        "results": results,
    }

    # Deliver webhook if provided
    if webhook_url:
        threading.Thread(
            target=_deliver_webhook,
            args=(webhook_url, response_data),
            daemon=True
        ).start()

    return jsonify(response_data), 200


@app.route("/compare", methods=["POST"])
def compare_analyses():
    data = request.get_json(silent=True) or {}
    id1 = data.get("analysis_id_1")
    id2 = data.get("analysis_id_2")

    if not id1 or not id2:
        return jsonify({"error": "Both analysis_id_1 and analysis_id_2 are required"}), 400

    path1 = os.path.join(RESULTS_DIR, f"{id1}.json")
    path2 = os.path.join(RESULTS_DIR, f"{id2}.json")

    if not os.path.exists(path1) or not os.path.exists(path2):
        return jsonify({"error": "One or both analyses not found"}), 404

    with open(path1, "r") as f:
        data1 = json.load(f)
    with open(path2, "r") as f:
        data2 = json.load(f)

    all_keywords_1 = set()
    all_keywords_2 = set()
    for r in data1.get("results", []):
        if "keywords" in r:
            all_keywords_1.update(r["keywords"])
    for r in data2.get("results", []):
        if "keywords" in r:
            all_keywords_2.update(r["keywords"])
    common_keywords = list(all_keywords_1 & all_keywords_2)

    sentiments_1 = [r.get("sentiment", "") for r in data1.get("results", [])]
    sentiments_2 = [r.get("sentiment", "") for r in data2.get("results", [])]
    min_len = min(len(sentiments_1), len(sentiments_2))
    matches = sum(1 for i in range(min_len) if sentiments_1[i] == sentiments_2[i])
    agreement = matches / min_len if min_len > 0 else 0.0

    chunk_delta = abs(data1.get("num_chunks", 0) - data2.get("num_chunks", 0))

    return jsonify({
        "common_keywords": common_keywords,
        "sentiment_agreement": agreement,
        "chunk_count_delta": chunk_delta,
    }), 200


@app.route("/results/<analysis_id>", methods=["GET"])
def get_results(analysis_id):
    filepath = os.path.join(RESULTS_DIR, f"{analysis_id}.json")
    if not os.path.exists(filepath):
        return jsonify({"error": "Analysis not found"}), 404

    with open(filepath, "r") as f:
        data = json.load(f)

    chunk_index = request.args.get("chunk_index", type=int)
    if chunk_index is not None:
        if chunk_index < 0 or chunk_index >= len(data["results"]):
            return jsonify({"error": "chunk_index out of bounds"}), 400
        data["results"] = [data["results"][chunk_index]]

    return jsonify(data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
