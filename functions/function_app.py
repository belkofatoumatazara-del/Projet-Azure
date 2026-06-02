"""
Event-driven ML pipeline — Azure Functions (Python v2 programming model).

Two functions in one app:

  dispatcher  (Event Grid trigger on Microsoft.Storage.BlobCreated for input/)
      Validates an uploaded CSV (extension, size, required columns). Valid files
      get a job message on the 'inference-jobs' queue; invalid files are moved
      to the rejected/ container with the reason in blob metadata.

  worker      (Queue trigger on 'inference-jobs')
      Downloads the CSV, calls the inference API for each row, writes the
      aggregated result to output/, and upserts one metadata record per row
      into Cosmos DB. Idempotent: deterministic document ids + upsert mean a
      message that is retried (or redelivered) never creates duplicates.

Resilience: the Storage Queue trigger retries automatically; after
maxDequeueCount (set to 5 in host.json) a message lands in the
'inference-jobs-poison' queue, which is our dead-letter queue.

All configuration comes from application settings (env vars) — nothing about
the deployment is hard-coded.
"""

import csv
import io
import json
import logging
import os
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import azure.functions as func
import requests
from azure.cosmos import CosmosClient
from azure.storage.blob import BlobServiceClient
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

app = func.FunctionApp()

# --- Configuration (application settings) -------------------------------------
API_URL = os.getenv("API_URL", "").rstrip("/")
COSMOS_CONN = os.getenv("COSMOS_CONNECTION", "")
COSMOS_DB = os.getenv("COSMOS_DB", "mlpipe")
COSMOS_CONTAINER = os.getenv("COSMOS_CONTAINER", "inferences")
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", str(10 * 1024 * 1024)))  # 10 MB

# The feature columns every uploaded CSV must contain (the model's contract).
REQUIRED_COLUMNS = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol",
]

# --- Lazily-initialised shared clients ----------------------------------------
# Created once per warm instance and reused across invocations (the "reuse an
# HttpClient" guidance) — cheaper than rebuilding connections each call.
_blob_service = None
_cosmos_container = None


def _session() -> requests.Session:
    if not hasattr(_session, "_s"):
        s = requests.Session()
        retry = Retry(total=3, backoff_factor=1,
                      status_forcelist=[502, 503, 504],
                      allowed_methods=["POST"])
        s.mount("https://", HTTPAdapter(max_retries=retry))
        _session._s = s
    return _session._s


def blob_service() -> BlobServiceClient:
    global _blob_service
    if _blob_service is None:
        _blob_service = BlobServiceClient.from_connection_string(
            os.environ["AzureWebJobsStorage"]
        )
    return _blob_service


def cosmos_container():
    global _cosmos_container
    if _cosmos_container is None and COSMOS_CONN:
        client = CosmosClient.from_connection_string(COSMOS_CONN)
        db = client.get_database_client(COSMOS_DB)
        _cosmos_container = db.get_container_client(COSMOS_CONTAINER)
    return _cosmos_container


def _safe_id(name: str) -> str:
    """Cosmos ids may not contain / \\ ? #."""
    for ch in "/\\?#":
        name = name.replace(ch, "_")
    return name


# --- Observability: custom metrics via OpenTelemetry -> Application Insights ---
# Three custom metrics required by the spec:
#   inference_count   (counter)   - number of inferences performed
#   model_latency_ms  (histogram) - end-to-end inference latency
#   api_error_count   (counter)   - failed calls to the inference API
# Initialised once per worker process; imports are inside the function so a
# missing package can never break function indexing/deployment.
_otel_ready = False
_instr: dict = {}


def _ensure_metrics() -> None:
    global _otel_ready
    if _otel_ready:
        return
    conn = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if not conn:
        return
    try:
        # Metrics-only setup: a dedicated MeterProvider with the Azure Monitor
        # metric exporter. This does NOT touch the worker's logging/tracing or
        # trace-context propagation, so it can't interfere with the host.
        from azure.monitor.opentelemetry.exporter import AzureMonitorMetricExporter
        from opentelemetry import metrics
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

        exporter = AzureMonitorMetricExporter.from_connection_string(conn)
        reader = PeriodicExportingMetricReader(exporter, export_interval_millis=15000)
        metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))
        meter = metrics.get_meter("mlpipe.worker")
        _instr["inferences"] = meter.create_counter(
            "inference_count", unit="1", description="Number of inferences performed")
        _instr["latency"] = meter.create_histogram(
            "model_latency_ms", unit="ms", description="End-to-end inference latency")
        _instr["errors"] = meter.create_counter(
            "api_error_count", unit="1", description="Inference API errors")
        _otel_ready = True
        logging.info("OpenTelemetry custom metrics initialised")
    except Exception as exc:  # noqa: BLE001
        logging.error("Could not initialise OTel metrics: %s", exc)


def _record_inference(latency_ms: float, label: str) -> None:
    if "inferences" in _instr:
        _instr["inferences"].add(1, {"label": label})
        _instr["latency"].record(latency_ms)


def _record_api_error() -> None:
    if "errors" in _instr:
        _instr["errors"].add(1)


def _flush_metrics() -> None:
    # On Consumption the worker may idle/stop between invocations, so force an
    # export at the end of each run to make sure the metrics actually ship.
    try:
        from opentelemetry import metrics
        metrics.get_meter_provider().force_flush(timeout_millis=5000)
    except Exception:  # noqa: BLE001
        pass


def _move_to_rejected(blob_name: str, reason: str) -> None:
    """Copy an invalid blob to rejected/ (download+upload, reliable for same
    account) with the reason recorded, then delete the original from input/."""
    logging.warning("Rejecting %s: %s", blob_name, reason)
    bs = blob_service()
    src = bs.get_blob_client("input", blob_name)
    try:
        data = src.download_blob().readall()
        bs.get_blob_client("rejected", blob_name).upload_blob(
            data, overwrite=True,
            metadata={"reject_reason": reason[:250].encode("ascii", "ignore").decode()},
        )
        src.delete_blob()
    except Exception as exc:  # noqa: BLE001
        logging.error("Failed to move %s to rejected/: %s", blob_name, exc)


# --- Dispatcher ---------------------------------------------------------------
@app.function_name(name="dispatcher")
@app.event_grid_trigger(arg_name="event")
@app.queue_output(arg_name="msg", queue_name="inference-jobs",
                  connection="AzureWebJobsStorage")
def dispatcher(event: func.EventGridEvent, msg: func.Out[str]) -> None:
    subject = event.subject or ""
    logging.info("Dispatcher event: %s", subject)

    # Only react to blobs created in the input/ container.
    if "/containers/input/blobs/" not in subject:
        logging.info("Ignoring event outside input/.")
        return
    blob_name = subject.split("/blobs/", 1)[1]

    # 1) Extension check.
    if not blob_name.lower().endswith(".csv"):
        _move_to_rejected(blob_name, "not a .csv file")
        return

    # 2) Size check (read properties before downloading the body).
    bc = blob_service().get_blob_client("input", blob_name)
    size = bc.get_blob_properties().size
    if size > MAX_FILE_BYTES:
        _move_to_rejected(blob_name, f"file too large: {size} bytes")
        return

    # 3) Schema check — header must contain every required feature column.
    try:
        text = bc.download_blob().readall().decode("utf-8-sig")
        header = next(csv.reader(io.StringIO(text)))
    except Exception as exc:  # noqa: BLE001
        _move_to_rejected(blob_name, f"unreadable CSV: {exc}")
        return
    header_set = {h.strip() for h in header}
    missing = [c for c in REQUIRED_COLUMNS if c not in header_set]
    if missing:
        _move_to_rejected(blob_name, f"missing columns: {missing}")
        return

    # Valid -> publish a job to the queue (decouples detection from processing).
    job = {
        "blob_name": blob_name,
        "container": "input",
        "size": size,
        "received_at": datetime.now(timezone.utc).isoformat(),
    }
    msg.set(json.dumps(job))
    logging.info("Enqueued job for %s (%d bytes)", blob_name, size)


# --- Worker -------------------------------------------------------------------
@app.function_name(name="worker")
@app.queue_trigger(arg_name="msg", queue_name="inference-jobs",
                   connection="AzureWebJobsStorage")
def worker(msg: func.QueueMessage) -> None:
    job = json.loads(msg.get_body().decode("utf-8"))
    blob_name = job["blob_name"]
    logging.info("Worker processing %s (dequeue #%d)", blob_name, msg.dequeue_count)

    if not API_URL:
        raise RuntimeError("API_URL is not configured")

    _ensure_metrics()
    bs = blob_service()
    text = bs.get_blob_client("input", blob_name).download_blob().readall().decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))

    container = cosmos_container()
    ts = datetime.now(timezone.utc)
    results = []

    for i, row in enumerate(reader):
        # Map spaced CSV headers -> the API's snake_case field names.
        payload = {col.replace(" ", "_"): float(row[col]) for col in REQUIRED_COLUMNS}

        t0 = time.perf_counter()
        try:
            resp = _session().post(f"{API_URL}/predict", json=payload, timeout=60)
            resp.raise_for_status()
        except Exception:
            _record_api_error()           # custom metric: api_error_count
            _flush_metrics()
            raise                          # -> retry -> poison queue (DLQ)
        out = resp.json()
        duration_ms = round((time.perf_counter() - t0) * 1000, 2)

        # custom metrics: inference_count + model_latency_ms
        _record_inference(duration_ms, out["label"])

        record = {
            "id": f"{_safe_id(blob_name)}-{i}",   # deterministic -> idempotent
            "file_name": blob_name,
            "row_index": i,
            "prediction": out["prediction"],
            "label": out["label"],
            "confidence": out["confidence"],
            "model_version": out["model_version"],
            "duration_ms": duration_ms,
            "timestamp": ts.isoformat(),
        }
        if container is not None:
            container.upsert_item(record)
        results.append(record)

    # Aggregated raw result to output/, named after the input blob + timestamp.
    stem = blob_name.rsplit(".", 1)[0]
    out_name = f"{stem}_{ts.strftime('%Y%m%dT%H%M%SZ')}.json"
    bs.get_blob_client("output", out_name).upload_blob(
        json.dumps({"source": blob_name, "count": len(results), "results": results},
                   indent=2),
        overwrite=True,
    )
    logging.info("Wrote %d result(s) to output/%s", len(results), out_name)
    _flush_metrics()


# --- Dashboard API (HTTP triggers) --------------------------------------------
# A Static Web App could not be used (its regions aren't in this subscription's
# allowed list), so the dashboard is served from the Function App instead:
#   GET /api/recent     -> 20 most recent inferences from Cosmos (JSON)
#   GET /api/dashboard  -> the dashboard HTML page
# Anonymous auth + a simple in-process rate limit of 60 requests/min per IP.

_RL_MAX = 60
_RL_WINDOW = 60.0
_rl_hits: dict = defaultdict(deque)


def _rate_ok(ip: str) -> bool:
    now = time.time()
    q = _rl_hits[ip]
    while q and now - q[0] > _RL_WINDOW:
        q.popleft()
    if len(q) >= _RL_MAX:
        return False
    q.append(now)
    return True


@app.function_name(name="recent")
@app.route(route="recent", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def recent(req: func.HttpRequest) -> func.HttpResponse:
    ip = (req.headers.get("X-Forwarded-For", "unknown").split(",")[0]).strip()
    if not _rate_ok(ip):
        return func.HttpResponse('{"error":"rate limit exceeded"}',
                                 status_code=429, mimetype="application/json")
    container = cosmos_container()
    if container is None:
        return func.HttpResponse('{"error":"cosmos not configured"}',
                                 status_code=503, mimetype="application/json")
    try:
        items = list(container.query_items(
            query=("SELECT TOP 20 c.file_name, c.row_index, c.prediction, c.label, "
                   "c.confidence, c.model_version, c.duration_ms, c.timestamp "
                   "FROM c ORDER BY c.timestamp DESC"),
            enable_cross_partition_query=True,
        ))
    except Exception as exc:  # noqa: BLE001
        logging.error("recent query failed: %s", exc)
        return func.HttpResponse(json.dumps({"error": "query failed", "detail": str(exc)[:300]}),
                                 status_code=500, mimetype="application/json")
    return func.HttpResponse(json.dumps(items), mimetype="application/json")


@app.function_name(name="dashboard")
@app.route(route="dashboard", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def dashboard(req: func.HttpRequest) -> func.HttpResponse:
    html = (Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8")
    return func.HttpResponse(html, mimetype="text/html")
