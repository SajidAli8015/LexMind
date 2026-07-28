"""
Shared HTTP helpers for LexMind CLI scripts.

Both ingest_cli.py and query_cli.py import from here so the
base URL, timeout, and error handling are defined in one place.
"""

import requests

REQUEST_TIMEOUT = 600  # seconds — ingestion of large PDFs can be slow

_UVICORN_HINT = (
    "Start the backend with:\n"
    "  uvicorn src.api.main:app --reload --port 8000"
)


class CLIAPIError(Exception):
    """Raised when an API call fails. Message includes the response body."""
    pass


def check_health(api_url: str) -> dict:
    """
    GET {api_url}/health

    Returns the parsed HealthResponse dict:
        status, version, documents_ingested, total_chunks

    Raises CLIAPIError with the uvicorn start command if the backend
    is unreachable, or if the response is not 200.
    """
    url = f"{api_url}/health"
    try:
        resp = requests.get(url, timeout=10)
    except requests.exceptions.ConnectionError:
        raise CLIAPIError(
            f"Cannot reach backend at {api_url}\n{_UVICORN_HINT}"
        )
    except requests.exceptions.Timeout:
        raise CLIAPIError(
            f"Health check timed out ({url})\n{_UVICORN_HINT}"
        )

    if resp.status_code != 200:
        raise CLIAPIError(
            f"GET /health returned {resp.status_code}: {resp.text}"
        )
    return resp.json()


def ingest_via_api(
    api_url: str,
    file_path: str,
    doc_title: str = None,
) -> dict:
    """
    POST {api_url}/api/ingest with the file as multipart/form-data.

    Sends:
        file      — the document file
        doc_title — optional title override (form field)

    Returns the parsed IngestResponse dict:
        success, doc_id, file_name, doc_title, chunks_created,
        articles_found, total_chars, message

    Raises CLIAPIError on connection failure or non-200 response,
    including the full response body so failures are diagnosable.
    """
    url = f"{api_url}/api/ingest"
    try:
        with open(file_path, "rb") as fh:
            files = {"file": (file_path, fh)}
            data = {}
            if doc_title:
                data["doc_title"] = doc_title
            resp = requests.post(
                url, files=files, data=data, timeout=REQUEST_TIMEOUT
            )
    except requests.exceptions.ConnectionError:
        raise CLIAPIError(
            f"Cannot reach backend at {api_url}\n{_UVICORN_HINT}"
        )
    except requests.exceptions.Timeout:
        raise CLIAPIError(
            f"POST /api/ingest timed out after {REQUEST_TIMEOUT}s"
        )

    if resp.status_code != 200:
        raise CLIAPIError(
            f"POST /api/ingest returned {resp.status_code}: {resp.text}"
        )
    return resp.json()


def query_via_api(
    api_url: str,
    query: str,
    doc_id: str = None,
) -> dict:
    """
    POST {api_url}/api/query with a JSON body.

    Sends:
        query  — the user's question (str)
        doc_id — optional document filter (str or None)

    Returns the parsed QueryResponse dict:
        success, query, query_type, final_answer, citations,
        groundedness_score, citation_score, relevance_score,
        critique_passed, regeneration_count, chunks_used, error

    Raises CLIAPIError on connection failure or non-200 response.
    """
    url = f"{api_url}/api/query"
    body = {"query": query}
    if doc_id:
        body["doc_id"] = doc_id

    try:
        resp = requests.post(url, json=body, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.ConnectionError:
        raise CLIAPIError(
            f"Cannot reach backend at {api_url}\n{_UVICORN_HINT}"
        )
    except requests.exceptions.Timeout:
        raise CLIAPIError(
            f"POST /api/query timed out after {REQUEST_TIMEOUT}s"
        )

    if resp.status_code != 200:
        raise CLIAPIError(
            f"POST /api/query returned {resp.status_code}: {resp.text}"
        )
    return resp.json()
