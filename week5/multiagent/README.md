# Single Agent vs Multi-Agent Token Consumption Benchmark

This project demonstrates and compares two distinct LLM architectures solving complex tasks that involve **live web search (via Tavily API)** and **exact mathematical calculations (via AST parser)**:

1. **Single Agent**: A single LLM (`qwen/qwen3.8-27b`) equipped with both search and math tools.
2. **Multi-Agent (Hierarchical Manager-Worker)**:
   - **Manager Agent** (`openai/gpt-oss-120b`): High-level task planner and orchestrator that delegates tasks and synthesizes final answers.
   - **Web Search Agent** (`openai/gpt-oss-20b`): Specialist sub-agent equipped with Tavily Search.
   - **Math Agent** (`qwen/qwen3.8-27b`): Specialist sub-agent equipped with the AST math calculator.

---

## 🛠️ Prerequisites & Setup

- Python 3.11 virtual environment (managed with `uv`).
- Environment variables configured in `.env`:
  ```env
  GROQ_API_KEY=gsk_...
  TAVILY_API_KEY=tvly-dev-...
  ```

To install dependencies using `uv`:
```bash
uv sync
```

---

## 🚀 How to Run

### 1. Interactive Menu
```bash
uv run python main.py
```

### 2. Run Single Agent Directly
```bash
uv run python single_agent.py
```
Or with custom query:
```bash
uv run python single_agent.py "Search for the latest GDP of Germany and Japan and calculate the difference"
```

### 3. Run Multi-Agent Directly
```bash
uv run python multi_agent.py
```
Or with custom query:
```bash
uv run python multi_agent.py "Search for the latest GDP of Germany and Japan and calculate the difference"
```

### 4. Run Head-to-Head Comparison Benchmark
```bash
uv run python compare.py
```

---

## 📊 Token Usage Metrics & Findings

Both architectures track prompt tokens, completion tokens, total tokens, and API calls per step.

### Benchmark Example Task:
> *"Search for the current population of Tokyo and the population of New York City, and calculate the difference between their populations and the percentage by which Tokyo is larger."*

| Metric | Single Agent | Multi-Agent (Combined 3 Models) |
| :--- | :--- | :--- |
| **Model(s)** | `qwen/qwen3.8-27b` | Manager: `gpt-oss-120b`<br>Search: `gpt-oss-20b`<br>Math: `qwen3.8-27b` |
| **API Calls** | 3 | 8 (Manager: 3, Search: 3, Math: 2) |
| **Prompt Tokens** | ~6,338 | ~7,207 |
| **Completion Tokens**| ~629 | ~1,339 |
| **Total Tokens** | **~6,967** | **~8,546** |
| **Execution Latency**| ~6.5 seconds | ~9.5 seconds |

### Key Architectural Takeaways:
1. **Modularity vs Token Overhead**: The Multi-Agent architecture consumes more tokens due to inter-agent communication (delegation prompts, context handoffs, and synthesis).
2. **Cost & Capability Optimization**: In a Multi-Agent architecture, you can offload heavy searching or math tasks to smaller, cost-effective models (20B / 27B) while keeping the expensive large model (120B) strictly for executive decision-making.
3. **Task Isolation & Resilience**: Specialized agents isolate errors and handle complex, multi-faceted workflows without overflowing a single model's prompt context.
