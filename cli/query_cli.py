"""
Query Pipeline CLI — step-by-step API-delegated debugger.

Delegates the full agent pipeline to the running FastAPI server
(ChromaDB vector queries crash the Python process on Windows outside
uvicorn) and presents each agent's output in a step-by-step format.

Optionally runs the Orchestrator locally first (--local-classify) for
comparison — this costs one extra LLM call and is therefore opt-in.

Usage:
    python -m cli.query_cli --query "What is the notice period?"
    python -m cli.query_cli --query "..." --doc-id contract_abc_a1b2c3d4
    python -m cli.query_cli --query "..." --no-pause
    python -m cli.query_cli --query "..." --show-prompt
    python -m cli.query_cli --query "..." --local-classify
    python -m cli.query_cli --query "..." --api-url http://localhost:8000
"""

import sys
import argparse
import time

# ── Rich setup with plain-print fallback ──────────────────────

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    _RICH = True
    console = Console()
except ImportError:
    _RICH = False

    class _FallbackConsole:
        def print(self, *args, **kwargs):
            import re
            text = " ".join(str(a) for a in args)
            text = re.sub(r'\[/?[a-zA-Z_ /:#0-9,]+\]', '', text)
            print(text)
        def rule(self, title="", **kwargs):
            w = 60
            if title:
                pad = max(0, (w - len(title) - 2) // 2)
                print("=" * pad + " " + title + " " + "=" * pad)
            else:
                print("=" * w)

    console = _FallbackConsole()


# ── Helpers ───────────────────────────────────────────────────

def header(title: str, step: str = None):
    if _RICH:
        label = f"{step} — {title}" if step else title
        console.print()
        console.rule(f"[bold cyan]{label}[/bold cyan]")
        console.print()
    else:
        label = f"{step} — {title}" if step else title
        console.print()
        console.rule(label)
        console.print()


def pause(no_pause: bool, next_name: str):
    if no_pause:
        return
    try:
        input(f"\n  [Enter] to continue to {next_name} > ")
    except EOFError:
        pass
    console.print()


def kv(label: str, value, color: str = "green"):
    if _RICH:
        console.print(f"  [bold]{label:<30}[/bold] [{color}]{value}[/{color}]")
    else:
        console.print(f"  {label:<30} {value}")


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


def score_fmt(val, threshold: float) -> tuple:
    """Returns (display_string, color) for a score value."""
    if val is None:
        return "not returned by API", "dim"
    mark = "✓" if val >= threshold else "✗"
    pct = int(val * 100)
    color = "green" if val >= threshold else "red"
    return f"{pct}%  {mark}  (threshold {int(threshold * 100)}%)", color


# ── Step functions ────────────────────────────────────────────

def step_0_health(api_url: str) -> dict:
    from cli.api_client import check_health, CLIAPIError
    try:
        health = check_health(api_url)
    except CLIAPIError as e:
        err(str(e))
        sys.exit(1)

    kv("API status",          health.get("status", "?"))
    kv("API version",         health.get("version", "?"))
    kv("Documents ingested",  health.get("documents_ingested", "?"))
    kv("Total chunks",        health.get("total_chunks", "?"))

    success("Backend is healthy")
    return health


def step_1_local_classify(query: str) -> dict:
    """Run the Orchestrator locally. Costs one extra LLM call."""
    warn("--local-classify: running Orchestrator locally (1 extra LLM call)")
    from src.graph.state import create_initial_state
    from src.agents.orchestrator import orchestrator_node

    state = create_initial_state(query=query)
    t0 = time.time()
    state = orchestrator_node(state)
    elapsed = time.time() - t0

    query_type = state.get("query_type", "unknown")
    query_intent = state.get("query_intent", "")

    kv("Local query type",   query_type, color="cyan")
    kv("Local query intent", query_intent)
    kv("Classify time",      f"{elapsed:.2f}s")

    if _RICH:
        console.print(
            f"\n  [dim]This classification is for inspection only. "
            f"The API will classify independently.[/dim]"
        )

    success(f"Local Orchestrator complete — '{query_type}'")
    return {"query_type": query_type, "query_intent": query_intent, "elapsed": elapsed}


def step_2_api_call(api_url: str, query: str, doc_id: str = None) -> tuple:
    from cli.api_client import query_via_api, CLIAPIError

    if _RICH:
        console.print("  [dim]Running 4-agent pipeline on server...[/dim]")
    else:
        console.print("  Running 4-agent pipeline on server...")

    t0 = time.time()
    try:
        resp = query_via_api(api_url, query, doc_id)
    except CLIAPIError as e:
        err(str(e))
        sys.exit(1)
    elapsed = time.time() - t0

    kv("API elapsed",  f"{elapsed:.2f}s")
    kv("API success",  resp.get("success", "?"))
    if resp.get("error"):
        warn(f"API error field: {resp['error']}")

    success("API call complete")
    return resp, elapsed


def step_3_retrieval(resp: dict):
    chunks_used = resp.get("chunks_used", None)

    if chunks_used is not None:
        kv("Chunks used by pipeline", chunks_used)
    else:
        kv("Chunks used", "not returned by API")

    if _RICH:
        console.print(
            "\n  [dim]Individual chunk texts and scores are not included in the\n"
            "  QueryResponse. Use POST /api/query via Swagger UI at\n"
            "  http://localhost:8000/docs to see the full agent state,\n"
            "  or add retrieved_chunks to QueryResponse in schemas.py.[/dim]"
        )
    else:
        console.print()
        console.print("  Individual chunk texts/scores: not returned by API.")
        console.print("  See http://localhost:8000/docs for the full agent state.")

    success(f"Retrieval step: {chunks_used or '?'} chunks used")


def step_4_reasoning(resp: dict, show_prompt: bool = False):
    final_answer = resp.get("final_answer", "")
    citations    = resp.get("citations") or []
    query_type   = resp.get("query_type", "not returned by API")

    kv("Server query type",   query_type, color="cyan")
    kv("Citations extracted", len(citations))

    section("Generated answer")
    if _RICH:
        console.print(Panel(
            final_answer or "[dim]No answer returned[/dim]",
            border_style="green",
            padding=(1, 2),
        ))
    else:
        console.print()
        console.print(final_answer or "  (no answer)")
        console.print()

    if citations:
        section("Extracted citations")
        for cit in citations:
            if _RICH:
                console.print(f"  • [cyan]{cit}[/cyan]")
            else:
                console.print(f"  - {cit}")

    if show_prompt:
        warn("--show-prompt: reasoning_prompt is not included in QueryResponse — "
             "not returned by API")

    success(f"Reasoning step: answer received, {len(citations)} citation(s)")


def step_5_critic(resp: dict):
    from src.config import settings

    g_score  = resp.get("groundedness_score")
    c_score  = resp.get("citation_score")
    r_score  = resp.get("relevance_score")
    passed   = resp.get("critique_passed")
    regen    = resp.get("regeneration_count", 0)

    g_str, g_col = score_fmt(g_score, settings.GROUNDEDNESS_THRESHOLD)
    c_str, c_col = score_fmt(c_score, settings.CITATION_THRESHOLD)
    r_str, r_col = score_fmt(r_score, settings.RELEVANCE_THRESHOLD)

    kv("Groundedness score", g_str, color=g_col)
    kv("Citation score",     c_str, color=c_col)
    kv("Relevance score",    r_str, color=r_col)

    if passed is None:
        kv("Critique passed", "not returned by API", color="dim")
    else:
        kv("Critique passed",
           "YES" if passed else "NO",
           color="green" if passed else "red")

    kv("Regeneration count", regen if regen is not None else "not returned by API")

    if regen and regen > 0:
        warn(f"Answer was regenerated {regen} time(s) before passing critique")

    success("Critic step complete")


def print_summary(query: str, resp: dict, timings: dict, local_classify: dict = None):
    from src.config import settings

    header("FINAL ANSWER")
    final_answer = resp.get("final_answer", "")
    if _RICH:
        console.print(Panel(
            final_answer or "[dim]No answer[/dim]",
            title="[bold green]Answer[/bold green]",
            border_style="green",
            padding=(1, 2),
        ))
    else:
        console.print()
        console.print(final_answer or "(no answer)")
        console.print()

    header("PIPELINE SUMMARY")
    g = resp.get("groundedness_score")
    c = resp.get("citation_score")
    r = resp.get("relevance_score")
    passed = resp.get("critique_passed")
    regen  = resp.get("regeneration_count", 0)

    def fmt(v):
        return f"{v:.2f}" if v is not None else "—"

    if _RICH:
        table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
        table.add_column("Field", style="bold", width=28)
        table.add_column("Value", style="green")

        table.add_row("Query",           query[:70] + ("…" if len(query) > 70 else ""))
        table.add_row("Query type (API)", resp.get("query_type", "—"))
        if local_classify:
            table.add_row("Query type (local)", local_classify.get("query_type", "—"))
        table.add_row("Chunks used",     str(resp.get("chunks_used", "—")))
        table.add_row("Citations",       str(len(resp.get("citations") or [])))
        table.add_row("Groundedness",    fmt(g))
        table.add_row("Citation score",  fmt(c))
        table.add_row("Relevance",       fmt(r))
        table.add_row("Critique passed", ("YES" if passed else "NO") if passed is not None else "—")
        table.add_row("Regenerations",   str(regen) if regen is not None else "—")
        table.add_row("", "")
        for name, t in timings.items():
            table.add_row(f"Time: {name}", f"{t:.2f}s")
        table.add_row("Total time", f"{sum(timings.values()):.2f}s")
        console.print(table)
    else:
        rows = [
            ("Query",            query[:70]),
            ("Query type (API)", resp.get("query_type", "—")),
        ]
        if local_classify:
            rows.append(("Query type (local)", local_classify.get("query_type", "—")))
        rows += [
            ("Chunks used",      str(resp.get("chunks_used", "—"))),
            ("Citations",        str(len(resp.get("citations") or []))),
            ("Groundedness",     fmt(g)),
            ("Citation score",   fmt(c)),
            ("Relevance",        fmt(r)),
            ("Critique passed",  ("YES" if passed else "NO") if passed is not None else "—"),
            ("Regenerations",    str(regen) if regen is not None else "—"),
        ] + [(f"Time: {n}", f"{t:.2f}s") for n, t in timings.items()] + \
            [("Total time", f"{sum(timings.values()):.2f}s")]
        for label, value in rows:
            console.print(f"  {label:<28} {value}")


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LexMind Query Pipeline — step-by-step API-delegated debugger",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m cli.query_cli --query "What is the notice period?"
  python -m cli.query_cli --query "Summarise the payment terms" --no-pause
  python -m cli.query_cli --query "What are the risks?" --show-prompt
  python -m cli.query_cli --query "..." --doc-id contract_abc_a1b2c3d4
  python -m cli.query_cli --query "..." --local-classify
  python -m cli.query_cli --query "..." --api-url http://localhost:8000
        """
    )
    parser.add_argument("--query",          required=True, help="Question to ask")
    parser.add_argument("--doc-id",         default=None,  help="Restrict to one document")
    parser.add_argument("--no-pause",       action="store_true", help="Run without pausing")
    parser.add_argument("--show-prompt",    action="store_true",
                        help="Print the LLM prompt if available (not returned by API)")
    parser.add_argument("--local-classify", action="store_true",
                        help="Run Orchestrator locally before API call (1 extra LLM call)")
    parser.add_argument("--api-url",        default="http://localhost:8000", metavar="URL",
                        help="Base API URL (default: http://localhost:8000)")
    args = parser.parse_args()

    if _RICH:
        console.print(Panel.fit(
            f"[bold cyan]LexMind Query Debugger[/bold cyan]\n"
            f"Query: [green]{args.query}[/green]\n"
            f"Doc ID: [yellow]{args.doc_id or 'all documents'}[/yellow]\n"
            f"API: [dim]{args.api_url}[/dim]\n"
            f"Local classify: [yellow]{args.local_classify}[/yellow]",
            border_style="cyan"
        ))
    else:
        console.print("=" * 60)
        console.print("  LexMind Query Debugger")
        console.print(f"  Query:          {args.query}")
        console.print(f"  Doc ID:         {args.doc_id or 'all documents'}")
        console.print(f"  API:            {args.api_url}")
        console.print(f"  Local classify: {args.local_classify}")
        console.print("=" * 60)

    timings = {}
    local_classify_result = None

    # ── STEP 0 — Health check ─────────────────────────────────
    header("HEALTH CHECK", step="STEP 0")
    step_0_health(args.api_url)
    pause(args.no_pause, "Step 1 — Orchestrator" if args.local_classify else "Step 2 — API call")

    # ── STEP 1 — Local Orchestrator (optional) ────────────────
    if args.local_classify:
        header("ORCHESTRATOR (local)", step="STEP 1")
        t0 = time.time()
        local_classify_result = step_1_local_classify(args.query)
        timings["1_local_classify"] = time.time() - t0
        pause(args.no_pause, "Step 2 — API call")

    # ── STEP 2 — API call ─────────────────────────────────────
    header("API CALL (4-agent pipeline)", step="STEP 2")
    resp, t = step_2_api_call(args.api_url, args.query, args.doc_id)
    timings["2_api_call"] = t
    pause(args.no_pause, "Step 3 — Retrieval results")

    # ── STEP 3 — Retrieval results ────────────────────────────
    header("RETRIEVAL RESULTS", step="STEP 3")
    step_3_retrieval(resp)
    pause(args.no_pause, "Step 4 — Reasoning result")

    # ── STEP 4 — Reasoning result ─────────────────────────────
    header("REASONING RESULT", step="STEP 4")
    step_4_reasoning(resp, show_prompt=args.show_prompt)
    pause(args.no_pause, "Step 5 — Critic scores")

    # ── STEP 5 — Critic scores ────────────────────────────────
    header("CRITIC SCORES", step="STEP 5")
    step_5_critic(resp)
    pause(args.no_pause, "Final Summary")

    # ── Final summary ─────────────────────────────────────────
    print_summary(args.query, resp, timings, local_classify=local_classify_result)

    if _RICH:
        console.print()
        console.rule("[bold green]QUERY COMPLETE[/bold green]")
    else:
        console.print()
        console.rule("QUERY COMPLETE")


if __name__ == "__main__":
    main()
