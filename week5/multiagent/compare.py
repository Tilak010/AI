import sys
import os
import time
from tabulate import tabulate

# Force UTF-8 on Windows terminal
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from single_agent import run_single_agent
from multi_agent import run_multi_agent


def run_benchmark(task: str):
    print("\n" + "=" * 80)
    print("🚀 [BENCHMARK] SINGLE AGENT vs MULTI-AGENT TOKEN COMPARISON")
    print(f"Task: {task}")
    print("=" * 80)

    # 1. Run Single Agent
    print("\n" + "#" * 60)
    print(">>> 1. RUNNING SINGLE AGENT (1 Model for both tasks)...")
    print("#" * 60)
    single_res = run_single_agent(task, verbose=True)

    # Small pause to allow API rate limits to reset comfortably
    time.sleep(2)

    # 2. Run Multi-Agent
    print("\n" + "#" * 60)
    print(">>> 2. RUNNING MULTI-AGENT (Manager + Search Agent + Math Agent)...")
    print("#" * 60)
    multi_res = run_multi_agent(task, verbose=True)

    # 3. Comparative Summary Table
    print("\n" + "=" * 80)
    print("📊 [HEAD-TO-HEAD TOKEN & PERFORMANCE COMPARISON]")
    print("=" * 80)

    s_tracker = single_res["tracker"]
    m_totals = multi_res["totals"]

    summary_rows = [
        [
            "Single Agent",
            single_res["model"],
            "Single LLM executes both Search & Math via tools",
            s_tracker.api_calls,
            s_tracker.prompt_tokens,
            s_tracker.completion_tokens,
            s_tracker.total_tokens,
            f"{single_res['elapsed_seconds']:.2f}s",
        ],
        [
            "Multi-Agent",
            "Manager: gpt-oss-120b\nSearch: gpt-oss-20b\nMath: qwen3.8-27b",
            "Specialized LLMs coordinated by an Executive Manager",
            m_totals["api_calls"],
            m_totals["prompt_tokens"],
            m_totals["completion_tokens"],
            m_totals["total_tokens"],
            f"{multi_res['elapsed_seconds']:.2f}s",
        ],
    ]

    headers = [
        "Architecture",
        "LLM(s) Employed",
        "Role / Delegation Style",
        "API Calls",
        "Prompt Tokens",
        "Completion Tokens",
        "Total Tokens",
        "Duration",
    ]
    print(tabulate(summary_rows, headers=headers, tablefmt="fancy_grid"))

    token_diff = m_totals["total_tokens"] - s_tracker.total_tokens
    token_diff_pct = (token_diff / s_tracker.total_tokens) * 100 if s_tracker.total_tokens else 0
    sign = "+" if token_diff >= 0 else ""

    print(f"\n💡 [ANALYSIS & TAKEAWAYS]:")
    print(f" • Token Overhead: Multi-Agent consumed {sign}{token_diff:,} tokens ({sign}{token_diff_pct:.1f}%) compared to Single Agent.")
    print(f" • Why: In Multi-Agent architectures, inter-agent communication (Manager prompts, sub-agent instructions, and synthesized reporting) creates modularity and specialization at the cost of additional prompt tokens.")
    print(f" • Advantage of Multi-Agent: Model specialization allows using smaller, cost-effective LLMs for specialized tasks (e.g. 20B for search, 27B for math) while reserving high-parameter models (120B) for orchestration.")
    print(f" • Advantage of Single Agent: Lower token overhead and single-context execution for linear tasks.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    benchmark_task = (
        "Search for the current population of Tokyo and the population of New York City, "
        "and calculate the difference between their populations and the percentage by which Tokyo is larger."
    )
    if len(sys.argv) > 1:
        benchmark_task = " ".join(sys.argv[1:])
    run_benchmark(benchmark_task)
