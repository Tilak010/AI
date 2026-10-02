import sys
import os
import json
import ast
import operator
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List
from dotenv import load_dotenv
from groq import Groq, BadRequestError
from tavily import TavilyClient
from tabulate import tabulate

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in environment or .env file.")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is not set in environment or .env file.")

groq_client = Groq(api_key=GROQ_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# Single Agent LLM (same model for both tasks)
SINGLE_AGENT_MODEL = "qwen/qwen3.8-27b"


# ============================================================
# TOKEN TRACKER
# ============================================================
@dataclass
class TokenTracker:
    name: str = "Single Agent"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    api_calls: int = 0
    step_details: List[Dict[str, Any]] = field(default_factory=list)

    def add_usage(self, usage, step_name: str = ""):
        if usage:
            pt = getattr(usage, "prompt_tokens", 0) or 0
            ct = getattr(usage, "completion_tokens", 0) or 0
            tt = getattr(usage, "total_tokens", 0) or (pt + ct)
            self.prompt_tokens += pt
            self.completion_tokens += ct
            self.total_tokens += tt
            self.api_calls += 1
            self.step_details.append({
                "step": step_name or f"Call #{self.api_calls}",
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": tt,
            })

    def summary_row(self) -> List[Any]:
        return [
            self.name,
            self.api_calls,
            self.prompt_tokens,
            self.completion_tokens,
            self.total_tokens,
        ]


# ============================================================
# HELPER: FLATTEN TOOL MESSAGES FOR SAFE SYNTHESIS
# ============================================================
def flatten_messages_for_text_summary(messages: list, final_prompt: str) -> list:
    system_text = "You are an expert AI assistant. Provide a direct, comprehensive summary based on the findings."
    collected = []
    for m in messages:
        if isinstance(m, dict):
            role = m.get("role")
            content = m.get("content", "")
            if role == "system" and content:
                system_text = content
            elif role == "user" and content:
                collected.append(f"Request: {content}")
            elif role == "tool" and content:
                collected.append(f"Retrieved Result:\n{content}")
        else:
            content = getattr(m, "content", None)
            if content:
                collected.append(f"Assistant: {content}")

    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": f"{chr(10).join(collected)}\n\nTask: {final_prompt}"}
    ]


# ============================================================
# TOOLS IMPLEMENTATION
# ============================================================
def web_search(query: str = "", cursor: int = 0, id: int = 0) -> str:
    """Search the web using Tavily for factual or real-time information."""
    effective_query = (query or "").strip()
    if not effective_query:
        effective_query = "latest statistics"
    try:
        response = tavily_client.search(query=effective_query, max_results=3)
        results = response.get("results", [])
        if not results:
            return f"No results found for query: '{effective_query}'"
        
        formatted = []
        for idx, item in enumerate(results, 1):
            title = item.get("title", "No Title")
            content = item.get("content", "No Content")
            url = item.get("url", "")
            formatted.append(f"[{idx}] {title}\nURL: {url}\nSnippet: {content}")
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Error executing web search: {str(e)}"


_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

def _eval_ast_node(node):
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)
        op_type = type(node.op)
        if op_type in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
    elif isinstance(node, ast.UnaryOp):
        operand = _eval_ast_node(node.operand)
        op_type = type(node.op)
        if op_type in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[op_type](operand)
        raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
    else:
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def calculate(expression: str) -> str:
    """Safely evaluates a mathematical expression string."""
    try:
        clean = (
            expression.strip()
            .replace(",", "")
            .replace("^", "**")
            .replace("x", "*")
            .replace("X", "*")
        )
        parsed = ast.parse(clean, mode="eval")
        result = _eval_ast_node(parsed.body)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression '{expression}': {str(e)}"


# Tools Schema
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Searches the live web using Tavily for factual data, numbers, statistics, and current information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query keywords"
                    },
                    "cursor": {
                        "type": "integer",
                        "description": "Pagination cursor offset"
                    },
                    "id": {
                        "type": "integer",
                        "description": "Document ID"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Performs exact mathematical calculations given an expression string, e.g. '(14000000 - 8336000)'. Parameter 'expression' is required.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate (e.g. '14000000 - 8336000')"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

TOOL_FUNCTIONS = {
    "web_search": web_search,
    "calculate": calculate
}


# ============================================================
# SINGLE AGENT EXECUTION
# ============================================================
def run_single_agent(user_query: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Executes a single agent powered by one LLM, handling both web search and calculations.
    Tracks and returns comprehensive token usage.
    """
    tracker = TokenTracker(name=f"Single Agent ({SINGLE_AGENT_MODEL})")
    start_time = time.time()

    if verbose:
        print("\n" + "=" * 70)
        print("🤖 [SINGLE AGENT SYSTEM]")
        print(f"   Architecture: Single LLM Agent with 2 Tools (Search + Calculate)")
        print(f"   Model:        {SINGLE_AGENT_MODEL}")
        print(f"   Task:         {user_query}")
        print("=" * 70)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an intelligent assistant equipped with two specialized tools:\n"
                "1. `web_search(query)`: Search the web using Tavily to retrieve current facts, numbers, or information.\n"
                "2. `calculate(expression)`: Perform exact mathematical calculations.\n\n"
                "Workflow Rules:\n"
                "- If you need factual information or numbers, call `web_search` with a clear query.\n"
                "- When you obtain the numerical data, call `calculate` with the mathematical expression to compute the answer.\n"
                "- Do NOT perform unnecessary repetitive searches.\n"
                "- After obtaining calculation results, provide a clear, comprehensive final answer to the user."
            )
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    max_steps = 6
    final_answer = ""
    step_history = []

    for step in range(1, max_steps + 1):
        if verbose:
            print(f"\n--- [Single Agent: Step {step}] ---")

        try:
            response = groq_client.chat.completions.create(
                model=SINGLE_AGENT_MODEL,
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto"
            )
            tracker.add_usage(response.usage, step_name=f"Step {step} (LLM Call)")
            msg = response.choices[0].message
        except BadRequestError:
            break

        tool_calls = msg.tool_calls

        if tool_calls:
            messages.append(msg)
            for tool_call in tool_calls:
                func_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except Exception:
                    args = {}

                if verbose:
                    print(f"   ⚙️  Tool Call: {func_name}")
                    print(f"      Arguments: {args}")

                func = TOOL_FUNCTIONS.get(func_name)
                if func:
                    result = func(**args)
                else:
                    result = f"Error: Tool '{func_name}' not found."

                preview = (str(result)[:140] + "...") if len(str(result)) > 140 else str(result)
                if verbose:
                    print(f"      Result:    {preview.replace(chr(10), ' ')}")

                step_history.append({
                    "step": step,
                    "tool": func_name,
                    "arguments": args,
                    "result": str(result)
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": str(result)
                })
        else:
            final_answer = msg.content or ""
            if verbose:
                print(f"   ✅ Final Answer Formulated")
            break

    if not final_answer:
        flat_msgs = flatten_messages_for_text_summary(
            messages,
            "Provide the final comprehensive answer based on the facts and calculations above."
        )
        resp_wrap = groq_client.chat.completions.create(
            model=SINGLE_AGENT_MODEL,
            messages=flat_msgs
        )
        tracker.add_usage(resp_wrap.usage, step_name="Final Synthesis")
        final_answer = resp_wrap.choices[0].message.content or ""

    elapsed = time.time() - start_time

    if verbose:
        print("\n" + "=" * 70)
        print("📋 [FINAL ANSWER]")
        print("=" * 70)
        print(final_answer.strip())
        print("\n" + "=" * 70)
        print("📊 [TOKEN CONSUMPTION REPORT - SINGLE AGENT]")
        print("=" * 70)
        
        detail_rows = [[d["step"], d["prompt_tokens"], d["completion_tokens"], d["total_tokens"]] for d in tracker.step_details]
        detail_headers = ["Step", "Prompt Tokens", "Completion Tokens", "Total Tokens"]
        print(tabulate(detail_rows, headers=detail_headers, tablefmt="simple"))
        print("-" * 50)
        
        headers = ["System / Model", "API Calls", "Prompt Tokens", "Completion Tokens", "Total Tokens"]
        print(tabulate([tracker.summary_row()], headers=headers, tablefmt="fancy_grid"))
        print(f"⏱️  Execution Time: {elapsed:.2f} seconds\n")

    return {
        "architecture": "Single Agent",
        "model": SINGLE_AGENT_MODEL,
        "final_answer": final_answer,
        "tracker": tracker,
        "step_history": step_history,
        "elapsed_seconds": elapsed,
    }


if __name__ == "__main__":
    task = (
        "Search for the current population of Tokyo and the population of New York City, "
        "and calculate the difference between their populations and the percentage by which Tokyo is larger."
    )
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    run_single_agent(task)
