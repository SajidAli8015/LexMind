"""
Ingestion Pipeline CLI — step-by-step debugger.

Runs each of the 5 ingestion stations individually so you can
inspect intermediate outputs and pause between steps.

Usage:
    python -m cli.ingest_cli --file path/to/doc.pdf
    python -m cli.ingest_cli --file path/to/doc.pdf --title "Labor Law"
    python -m cli.ingest_cli --file path/to/doc.pdf --no-pause
    python -m cli.ingest_cli --file path/to/doc.pdf --station 2
"""

import sys
import argparse
import time
from pathlib import Path

# ── Rich setup with plain-print fallback ──────────────────────

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.rule import Rule
    from rich import box
    from rich.text import Text
    _RICH = True
    console = Console()
except ImportError:
    _RICH = False

    class _FallbackConsole:
        def print(self, *args, **kwargs):
            # Strip rich markup tags for plain output
            import re
            text = " ".join(str(a) for a in args)
            text = re.sub(r'\[/?[a-zA-Z_ /:#0-9,]+\]', '', text)
            print(text)
        def rule(self, title="", **kwargs):
            w = 60
            if title:
                pad = (w - len(title) - 2) // 2
                print("=" * pad + " " + title + " " + "=" * pad)
            else:
                print("=" * w)

    console = _FallbackConsole()


# ── Helpers ───────────────────────────────────────────────────

def header(title: str, station: int = None, total: int = 5):
    if _RICH:
        label = f"STATION {station}/{total} — {title}" if station else title
        console.print()
        console.rule(f"[bold cyan]{label}[/bold cyan]")
        console.print()
    else:
        label = f"STATION {station}/{total} — {title}" if station else title
        console.print()
        console.rule(label)
        console.print()


def pause(no_pause: bool, station_name: str):
    if no_pause:
        return
    try:
        input(f"\n  [Enter] to continue to {station_name} > ")
    except EOFError:
        pass
    console.print()


def info(msg: str):
    if _RICH:
        console.print(f"  [dim]{msg}[/dim]")
    else:
        console.print(f"  {msg}")


def kv(label: str, value, color: str = "green"):
    if _RICH:
        console.print(f"  [bold]{label:<28}[/bold] [{color}]{value}[/{color}]")
    else:
        console.print(f"  {label:<28} {value}")


def section(title: str):
    if _RICH:
        console.print(f"\n  [bold yellow]{title}[/bold yellow]")
    else:
        console.print(f"\n  -- {title} --")


def success(msg: str):
    if _RICH:
        console.print(f"\n[bold green]  ✓ {msg}[/bold green]")
    else:
        console.print(f"\n  OK: {msg}")


def warn(msg: str):
    if _RICH:
        console.print(f"  [bold yellow]! {msg}[/bold yellow]")
    else:
        console.print(f"  WARN: {msg}")


def err(msg: str):
    if _RICH:
        console.print(f"  [bold red]✗ {msg}[/bold red]")
    else:
        console.print(f"  ERROR: {msg}")


# ── Station functions ─────────────────────────────────────────

def run_station_1(file_path: str) -> "ParsedDocument":
    from src.ingestion.parser import DocumentParser

    t0 = time.time()
    parser = DocumentParser()
    parsed = parser.parse(file_path)
    elapsed = time.time() - t0

    elements = parsed.elements
    type_counts = {}
    level_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for e in elements:
        type_counts[e.element_type] = type_counts.get(e.element_type, 0) + 1
        if e.level in level_counts:
            level_counts[e.level] += 1

    kv("Parser used",   parsed.metadata.get("parser", "unknown"))
    kv("Total pages",   parsed.total_pages)
    kv("Total elements", len(elements))
    kv("Parse time",    f"{elapsed:.2f}s")

    section("Element type breakdown")
    for etype, count in sorted(type_counts.items()):
        kv(f"  {etype}", count)

    section("Heading level breakdown")
    level_labels = {0: "paragraph (L0)", 1: "part/chapter (L1)",
                    2: "article/section (L2)", 3: "sub-clause (L3)"}
    for lvl, count in level_counts.items():
        kv(f"  {level_labels[lvl]}", count)

    section("First 10 elements")
    for i, e in enumerate(elements[:10], 1):
        preview = e.text[:80].replace("\n", " ")
        if _RICH:
            console.print(
                f"  [dim]{i}.[/dim] "
                f"[cyan][{e.element_type.upper():9} L{e.level}][/cyan] "
                f"{preview}"
            )
        else:
            console.print(f"  {i}. [{e.element_type.upper():9} L{e.level}] {preview}")

    success(f"Station 1 complete — {len(elements)} elements extracted")
    return parsed, elapsed


def run_station_2(parsed, doc_title: str = None) -> tuple:
    from src.ingestion.chunker import LegalChunker
    from src.ingestion.title_extractor import detect_document_title

    t0 = time.time()
    chunker = LegalChunker()
    chunking_result = chunker.chunk(parsed)
    elapsed = time.time() - t0

    # Detect title
    try:
        from src.llm_client import get_llm
        llm = get_llm()
    except Exception:
        llm = None

    detected_title = detect_document_title(
        elements=parsed.elements,
        file_name=parsed.file_name,
        llm=llm,
        user_provided=doc_title,
    )

    chunks = chunking_result.chunks
    sizes = [c.char_count for c in chunks] if chunks else [0]

    kv("Doc ID",          chunking_result.doc_id)
    kv("Detected title",  detected_title)
    kv("Total chunks",    chunking_result.total_chunks)
    kv("Article chunks",  chunking_result.articles_found)
    kv("Total characters", chunking_result.total_chars)
    kv("Average chunk",   f"{sum(sizes) // len(sizes) if sizes else 0} chars")
    kv("Largest chunk",   f"{max(sizes)} chars")
    kv("Smallest chunk",  f"{min(sizes)} chars")
    kv("Chunk time",      f"{elapsed:.2f}s")

    # Stamp metadata
    for chunk in chunks:
        chunk.metadata["doc_title"] = detected_title
        chunk.metadata["file_name"] = parsed.file_name

    section("First 10 chunks")
    for chunk in chunks[:10]:
        preview = chunk.text[:200].replace("\n", " ")
        if _RICH:
            console.print(
                f"  [bold]Chunk {chunk.chunk_index}[/bold] "
                f"[cyan][{chunk.chunk_type.upper()}][/cyan] "
                f"[dim]article_ref=[/dim][green]{chunk.article_ref}[/green] "
                f"[dim]({chunk.char_count} chars)[/dim]"
            )
            console.print(f"    [dim]{preview}…[/dim]")
        else:
            console.print(f"  Chunk {chunk.chunk_index} [{chunk.chunk_type.upper()}] "
                          f"article_ref={chunk.article_ref} ({chunk.char_count} chars)")
            console.print(f"    {preview}...")
        console.print()

    success(f"Station 2 complete — {len(chunks)} chunks created, title='{detected_title}'")
    return chunking_result, detected_title, elapsed


def run_station_3(chunking_result) -> tuple:
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["ONNXRUNTIME_DISABLE_EXCEPTIONS"] = "0"

    from src.ingestion.embedder import DocumentEmbedder

    info("Loading E5 model — first run downloads ~1.3 GB, please wait...")
    t0 = time.time()
    try:
        embedder = DocumentEmbedder()
        embedder._load_model()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nEmbedder failed: {e}")
        print("\nTry running: pip install onnxruntime==1.19.2 safetensors==0.4.5")
        sys.exit(1)
    embedding_result = embedder.embed_chunks(chunking_result)
    elapsed = time.time() - t0

    chunks = embedding_result.embedded_chunks
    all_dims = [c.embedding_dim for c in chunks]
    consistent = len(set(all_dims)) == 1

    kv("Vectors created",   embedding_result.total_chunks)
    kv("Embedding dim",     embedding_result.embedding_dim)
    kv("Embedding model",   embedding_result.model_name)
    kv("Embed time",        f"{elapsed:.2f}s")
    kv("Dims consistent",   "YES" if consistent else "NO — mismatch!")

    if chunks:
        section("First vector — first 8 values")
        sample = chunks[0].embedding[:8]
        formatted = "  [" + ", ".join(f"{v:.6f}" for v in sample) + ", ...]"
        if _RICH:
            console.print(f"  [cyan]{formatted}[/cyan]")
        else:
            console.print(formatted)

    if not consistent:
        warn("Embedding dimensions are inconsistent across chunks!")

    success(f"Station 3 complete — {len(chunks)} vectors at dim={embedding_result.embedding_dim}")
    return embedding_result, elapsed


def station_4_store(chunks):
    """
    Station 4: Vector Store — read-only verification mode.

    The CLI runs in the same process as the Windows-incompatible
    ChromaDB native writer. Instead of re-writing, we verify what
    is already stored and show the metadata structure.
    """
    from src.ingestion.vector_store import VectorStore

    store = VectorStore()
    collection = store._get_collection()
    total = collection.count()

    print(f"\n  ChromaDB path: ./data/chroma_db")
    print(f"  Total chunks in collection: {total}")
    print(f"  Chunks this document would write: {len(chunks)}")
    print()

    # Show what metadata would be stored per chunk
    print("  Metadata structure stored per chunk:")
    sample = chunks[0]
    print(f"    doc_id:      {sample.doc_id}")
    print(f"    article_ref: {sample.article_ref}")
    print(f"    chunk_type:  {sample.chunk_type}")
    print(f"    chunk_index: {sample.chunk_index}")
    print(f"    char_count:  {sample.char_count}")
    print(f"    doc_title:   {sample.metadata.get('doc_title', 'N/A')}")
    print(f"    file_name:   {sample.metadata.get('file_name', 'N/A')}")
    print()

    print("  Vector search verification: skipped in CLI mode")
    print("    (ChromaDB native HNSW query is not stable outside")
    print("     the uvicorn server process on Windows)")

    print()
    print("  NOTE: Write skipped in CLI mode to avoid Windows native")
    print("        memory conflict with ChromaDB SQLite writer.")
    print("        Use the API (POST /api/ingest) for actual ingestion.")
    print()
    print("  [STATION 4 COMPLETE — read-only verification]")

    return total


def run_station_5() -> tuple:
    from src.config import settings

    t0 = time.time()
    indexed_count = 0
    index_path = settings.BM25_INDEX_PATH
    error_msg = None

    try:
        from src.ingestion.vector_store import VectorStore
        from src.agents.retrieval_agent import BM25Index

        store = VectorStore()          # create locally, do not rely on arg
        bm25 = BM25Index()
        indexed_count = bm25.build_from_vector_store(store)
    except Exception as e:
        error_msg = str(e)

    elapsed = time.time() - t0

    kv("Chunks indexed", indexed_count if not error_msg else "FAILED")
    kv("Index path", index_path)

    index_file = Path(index_path)
    if index_file.exists():
        size_kb = index_file.stat().st_size / 1024
        kv("Index file size", f"{size_kb:.1f} KB")
    else:
        kv("Index file size", "file not found")

    kv("Rebuild time", f"{elapsed:.2f}s")

    if error_msg:
        warn(f"BM25 rebuild failed (non-fatal): {error_msg}")
        warn("Dense search still works — BM25 is optional")
    else:
        success(f"Station 5 complete — {indexed_count} chunks indexed in BM25")

    return indexed_count, elapsed, error_msg


# ── Summary table ─────────────────────────────────────────────

def print_summary(
    file_path, doc_id, doc_title, chunks_created, articles_found,
    total_chars, embedding_dim, timings
):
    header("FINAL SUMMARY")
    if _RICH:
        table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
        table.add_column("Field", style="bold", width=24)
        table.add_column("Value", style="green")

        table.add_row("File",            Path(file_path).name)
        table.add_row("Doc ID",          doc_id or "—")
        table.add_row("Doc title",        doc_title or "—")
        table.add_row("Chunks created",   str(chunks_created))
        table.add_row("Articles found",   str(articles_found))
        table.add_row("Total characters", f"{total_chars:,}")
        table.add_row("Embedding dim",    str(embedding_dim))
        table.add_row("", "")
        for name, t in timings.items():
            table.add_row(f"Time: {name}", f"{t:.2f}s")
        table.add_row("Total time", f"{sum(timings.values()):.2f}s")

        console.print(table)
    else:
        rows = [
            ("File",             Path(file_path).name),
            ("Doc ID",           doc_id or "-"),
            ("Doc title",        doc_title or "-"),
            ("Chunks created",   str(chunks_created)),
            ("Articles found",   str(articles_found)),
            ("Total characters", f"{total_chars:,}"),
            ("Embedding dim",    str(embedding_dim)),
        ] + [(f"Time: {n}", f"{t:.2f}s") for n, t in timings.items()] + \
            [("Total time", f"{sum(timings.values()):.2f}s")]
        for label, value in rows:
            console.print(f"  {label:<28} {value}")


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LexMind Ingestion Pipeline — step-by-step debugger",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m cli.ingest_cli --file docs/labor_law.pdf
  python -m cli.ingest_cli --file docs/labor_law.pdf --title "Saudi Labor Law"
  python -m cli.ingest_cli --file docs/labor_law.pdf --no-pause
  python -m cli.ingest_cli --file docs/labor_law.pdf --station 2
        """
    )
    parser.add_argument("--file",     required=True, help="Path to document to ingest")
    parser.add_argument("--title",    default=None,  help="Document title override")
    parser.add_argument("--no-pause", action="store_true", help="Run without pausing")
    parser.add_argument("--station",  type=int, default=5, metavar="N",
                        help="Run only up to station N then stop (1-5)")
    parser.add_argument("--write",    action="store_true",
                        help="Actually ingest by calling POST /api/ingest (stations 4-5 via API)")
    parser.add_argument("--api-url",  default="http://localhost:8000", metavar="URL",
                        help="Base API URL (default: http://localhost:8000)")
    args = parser.parse_args()

    file_path = args.file
    if not Path(file_path).exists():
        err(f"File not found: {file_path}")
        sys.exit(1)

    mode_label = "API write mode (--write)" if args.write else "read-only / inspect mode"
    if _RICH:
        console.print(Panel.fit(
            f"[bold cyan]LexMind Ingestion Debugger[/bold cyan]\n"
            f"File: [green]{file_path}[/green]\n"
            f"Title override: [yellow]{args.title or 'auto-detect'}[/yellow]\n"
            f"Run to station: [yellow]{args.station}[/yellow]\n"
            f"Mode: [yellow]{mode_label}[/yellow]"
            + (f"\nAPI: [dim]{args.api_url}[/dim]" if args.write else ""),
            border_style="cyan"
        ))
    else:
        console.print("=" * 60)
        console.print("  LexMind Ingestion Debugger")
        console.print(f"  File: {file_path}")
        console.print(f"  Title: {args.title or 'auto-detect'}")
        console.print(f"  Run to station: {args.station}")
        console.print(f"  Mode: {mode_label}")
        if args.write:
            console.print(f"  API: {args.api_url}")
        console.print("=" * 60)

    # Health check upfront when --write is set
    if args.write:
        from cli.api_client import check_health, CLIAPIError
        try:
            health = check_health(args.api_url)
            info(f"Backend healthy — {health.get('documents_ingested', '?')} docs, "
                 f"{health.get('total_chunks', '?')} chunks in store")
        except CLIAPIError as e:
            err(str(e))
            sys.exit(1)

    timings = {}
    doc_id = None
    doc_title = None
    chunks_created = 0
    articles_found = 0
    total_chars = 0
    embedding_dim = 0
    parsed = None
    chunking_result = None
    embedding_result = None


    # ── Station 1 ────────────────────────────────────────────
    header("PARSER", station=1)
    parsed, t = run_station_1(file_path)
    timings["1_parse"] = t
    if args.station < 2:
        console.print("\n  Stopped at station 1 (--station 1)")
        return
    pause(args.no_pause, "Station 2 — Chunker")

    # ── Station 2 ────────────────────────────────────────────
    header("CHUNKER + TITLE DETECTION", station=2)
    chunking_result, doc_title, t = run_station_2(parsed, args.title)
    doc_id = chunking_result.doc_id
    articles_found = chunking_result.articles_found
    total_chars = chunking_result.total_chars
    timings["2_chunk"] = t
    if args.station < 3:
        console.print("\n  Stopped at station 2 (--station 2)")
        return
    pause(args.no_pause, "Station 3 — Embedder")

    # ── Station 3 ────────────────────────────────────────────
    header("EMBEDDER (E5-large)", station=3)
    embedding_result, t = run_station_3(chunking_result)
    chunks_created = embedding_result.total_chunks
    embedding_dim = embedding_result.embedding_dim
    timings["3_embed"] = t
    if args.station < 4:
        console.print("\n  Stopped at station 3 (--station 3)")
        return
    pause(args.no_pause, "Station 4 — Vector Store")

    # ── Station 4 ────────────────────────────────────────────
    header("VECTOR STORE (ChromaDB)", station=4)
    t0 = time.time()

    if args.write:
        # Delegate to API — ChromaDB native layer is only stable inside uvicorn
        if _RICH:
            console.print(
                "  [bold yellow]Delegating stations 4-5 to the API server.[/bold yellow]\n"
                "  [dim]ChromaDB's native HNSW library is not stable outside the\n"
                "  uvicorn server process on Windows. The API handles the write.[/dim]\n"
            )
        else:
            console.print("  Delegating stations 4-5 to the API server.")
            console.print("  (ChromaDB native HNSW is not stable outside uvicorn on Windows)")
            console.print()

        from cli.api_client import ingest_via_api, CLIAPIError
        try:
            api_resp = ingest_via_api(args.api_url, file_path, args.title)
        except CLIAPIError as e:
            err(str(e))
            sys.exit(1)

        kv("API doc_id",        api_resp.get("doc_id", "—"))
        kv("API doc_title",     api_resp.get("doc_title", "—"))
        kv("API chunks_created", api_resp.get("chunks_created", "—"))
        kv("API articles_found", api_resp.get("articles_found", "—"))
        kv("API total_chars",   f"{api_resp.get('total_chars', 0):,}")
        kv("API message",       api_resp.get("message", "—"))

        api_chunks = api_resp.get("chunks_created", 0)
        local_chunks = len(chunking_result.chunks)
        match = api_chunks == local_chunks
        kv("Chunk count match",
           f"YES (local {local_chunks} == API {api_chunks})" if match
           else f"NO — local={local_chunks} API={api_chunks} (sub-chunks may differ)",
           color="green" if match else "yellow")

        # Update summary fields from API response
        doc_id        = api_resp.get("doc_id", doc_id)
        doc_title     = api_resp.get("doc_title", doc_title)
        chunks_created = api_resp.get("chunks_created", chunks_created)
        articles_found = api_resp.get("articles_found", articles_found)
        total_chars   = api_resp.get("total_chars", total_chars)

        success("Station 4 complete — document written via API")
    else:
        station_4_store(chunking_result.chunks)
        chunks_created = len(chunking_result.chunks)

    t = time.time() - t0
    timings["4_store"] = t
    if args.station < 5:
        console.print("\n  Stopped at station 4 (--station 4)")
        return
    pause(args.no_pause, "Station 5 — BM25 Rebuild")

    # ── Station 5 ────────────────────────────────────────────
    header("BM25 INDEX REBUILD", station=5)

    if args.write:
        # API already rebuilt BM25 as part of ingestion (Station 5 in pipeline)
        from src.config import settings
        t0 = time.time()
        info("BM25 index was rebuilt by the API during ingestion.")
        index_file = Path(settings.BM25_INDEX_PATH)
        if index_file.exists():
            import datetime
            size_kb = index_file.stat().st_size / 1024
            mtime = datetime.datetime.fromtimestamp(index_file.stat().st_mtime)
            kv("Index path",      settings.BM25_INDEX_PATH)
            kv("Index file size", f"{size_kb:.1f} KB")
            kv("Last modified",   mtime.strftime("%Y-%m-%d %H:%M:%S"))
            success(f"Station 5 complete — BM25 index present at {size_kb:.1f} KB")
        else:
            warn(f"BM25 index not found at {settings.BM25_INDEX_PATH}")
        t = time.time() - t0
        timings["5_bm25"] = t
    else:
        _, t, _ = run_station_5()
        timings["5_bm25"] = t

    # ── Summary ───────────────────────────────────────────────
    print_summary(
        file_path=file_path,
        doc_id=doc_id,
        doc_title=doc_title,
        chunks_created=chunks_created,
        articles_found=articles_found,
        total_chars=total_chars,
        embedding_dim=embedding_dim,
        timings=timings,
    )

    if _RICH:
        console.print()
        console.rule("[bold green]INGESTION COMPLETE[/bold green]")
    else:
        console.print()
        console.rule("INGESTION COMPLETE")


if __name__ == "__main__":
    main()
