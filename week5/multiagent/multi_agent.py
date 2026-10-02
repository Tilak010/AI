import sys
import os
import json
import ast
import operator
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from groq import Groq, BadRequestError
from tavily import TavilyClient
from tabulate import tabulate

# Force UTF-8 encoding on Windows terminal
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

# ============================================================
# MULTI-AGENT MODEL CONFIGURATION
# Each role uses a DIFFERENT LLM
# ============================================================
MANAGER_MODEL = "openai/gpt-oss-120b"     # Orchestrator & Planner (120B reasoning model)
SEARCH_AGENT_MODEL = "openai/gpt-oss-20b" # Specialist for Web Search & Information Retrieval (20B model)
MATH_AGENT_MODEL = "qwen/qwen3.8-27b"     # Specialist for Numerical & Mathematical Evaluation (27B model)


# ============================================================
# TOKEN TRACKER
# ============================================================
@dataclass
class TokenTracker:
    name: str
    model: str
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
            self.model,
            self.api_calls,
            self.prompt_tokens,
            self.completion_tokens,
            self.total_tokens,
        ]


# ============================================================
# HELPER: FLATTEN TOOL MESSAGES FOR SAFE TEXT SYNTHESIS
# Prevents Groq 'Tool choice is none, but model called a tool' errors
# ============================================================
def flatten_messages_for_text_summary(messages: list, final_prompt: str) -> list:
    """
    Extracts all retrieved facts and system instructions into pure text messages
    without raw tool-call objects, allowing the model to summarize cleanly.
    """
    system_text = "You are an expert AI assistant. Provide a direct, comprehensive summary based on the provided findings."
    collected_facts = []

    for m in messages:
        if isinstance(m, dict):
            role = m.get("role")
            content = m.get("content", "")
            if role == "system" and content:
                system_text = content
            elif role == "user" and content:
                collected_facts.append(f"Request: {content}")
            elif role == "tool" and content:
                collected_facts.append(f"Retrieved Findings:\n{content}")
        else:
            # ChatCompletionMessage object
            content = getattr(m, "content", None)
            if content:
                collected_facts.append(f"Assistant Note: {content}")

    context_str = "\n\n".join(collected_facts)
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": f"{context_str}\n\nTask: {final_prompt}"}
    ]


# ============================================================
# LOW-LEVEL ENGINE TOOLS
# ============================================================
def execute_web_search(query: Optional[str] = None, cursor: int = 0, id: int = 0, fallback_context: str = "") -> str:
    """Execute live web search via Tavily API with pagination and fallback support."""
    effective_query = (query or "").strip()
    if not effective_query:
        effective_query = fallback_context.strip()
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


def execute_calculate(expression: str) -> str:
    """Safely evaluates arithmetic expressions."""
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


# ============================================================
# SPECIALIZED AGENT 1: WEB SEARCH AGENT
# Model: SEARCH_AGENT_MODEL (openai/gpt-oss-20b)
# ============================================================
SEARCH_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Searches the live web via Tavily for real-time information, statistics, and facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query keywords"
                    },
                    "cursor": {
                        "type": "integer",
                        "description": "Pagination cursor offset"
                    },
                    "id": {
                        "type": "integer",
                        "description": "Result document ID"
                    }
                }
            }
        }
    }
]

def run_search_agent(instruction: str, tracker: TokenTracker, verbose: bool = True) -> str:
    """
    Sub-agent dedicated to searching and extracting facts.
    Uses SEARCH_AGENT_MODEL (openai/gpt-oss-20b).
    """
    if verbose:
        print(f"\n   🔎 [Search Agent | Model: {SEARCH_AGENT_MODEL}]")
        print(f"      Task: '{instruction}'")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a dedicated Web Research Specialist Agent.\n"
                "Your objective is to find accurate factual information, statistics, and figures.\n"
                "You have access to the `web_search(query)` tool.\n"
                "Once you obtain search results, summarize the essential data points and numbers in your response text."
            )
        },
        {
            "role": "user",
            "content": f"Please research and provide exact figures/facts for: {instruction}"
        }
    ]

    max_search_turns = 3

    for step in range(max_search_turns):
        try:
            response = groq_client.chat.completions.create(
                model=SEARCH_AGENT_MODEL,
                messages=messages,
                tools=SEARCH_TOOL_SCHEMA,
                tool_choice="auto"
            )
            tracker.add_usage(response.usage, step_name=f"Search Agent Step {step+1}")
            msg = response.choices[0].message
        except BadRequestError:
            # Handle tool schema validation or unexpected token generation
            break

        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}
                query = args.get("query")
                cursor = args.get("cursor", 0)
                doc_id = args.get("id", 0)
                
                search_res = execute_web_search(query=query, cursor=cursor, id=doc_id, fallback_context=instruction)
                if verbose:
                    q_display = query or instruction
                    print(f"      🌐 Tavily Search: '{q_display}'")
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": search_res
                })
        else:
            ans = msg.content or ""
            if verbose:
                preview = (ans[:140] + "...") if len(ans) > 140 else ans
                print(f"      📋 Search Agent Findings: {preview.replace(chr(10), ' ')}")
            return ans

    # Safe text summary without raw tool tags to prevent 'Tool choice is none' error
    try:
        flat_messages = flatten_messages_for_text_summary(
            messages,
            "Please provide a concise, factual summary of the key numbers and data points found above."
        )
        final_resp = groq_client.chat.completions.create(
            model=SEARCH_AGENT_MODEL,
            messages=flat_messages
        )
        tracker.add_usage(final_resp.usage, step_name="Search Agent Summary")
        summary_ans = final_resp.choices[0].message.content or ""
        if verbose:
            preview = (summary_ans[:140] + "...") if len(summary_ans) > 140 else summary_ans
            print(f"      📋 Search Agent Findings: {preview.replace(chr(10), ' ')}")
        return summary_ans
    except Exception:
        # Fallback: extract tool content strings directly
        tool_outputs = [m["content"] for m in messages if isinstance(m, dict) and m.get("role") == "tool"]
        return "\n\n".join(tool_outputs[:2]) if tool_outputs else "Search data retrieved."


# ============================================================
# SPECIALIZED AGENT 2: MATH / CALCULATOR AGENT
# Model: MATH_AGENT_MODEL (qwen/qwen3.8-27b)
# ============================================================
MATH_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Calculates the exact result of a mathematical expression. Parameter 'expression' is required.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression, e.g. '(14192184 - 8478000) / 8478000 * 100'"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

def run_math_agent(instruction: str, tracker: TokenTracker, verbose: bool = True) -> str:
    """
    Sub-agent dedicated to arithmetic computations and mathematical verification.
    Uses MATH_AGENT_MODEL (qwen/qwen3.8-27b).
    """
    if verbose:
        print(f"\n   🔢 [Math Agent | Model: {MATH_AGENT_MODEL}]")
        print(f"      Task: '{instruction}'")

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert Mathematical Calculator Agent.\n"
                "Your objective is to solve the given math computation accurately.\n"
                "You have access to the `calculate(expression)` tool.\n"
                "Always invoke the `calculate` tool to compute exact values rather than guessing.\n"
                "Return the computed results with step-by-step clarity."
            )
        },
        {
            "role": "user",
            "content": instruction
        }
    ]

    for step in range(3):
        try:
            response = groq_client.chat.completions.create(
                model=MATH_AGENT_MODEL,
                messages=messages,
                tools=MATH_TOOL_SCHEMA,
                tool_choice="auto"
            )
            tracker.add_usage(response.usage, step_name=f"Math Agent Step {step+1}")
            msg = response.choices[0].message
        except BadRequestError:
            break

        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}
                expr = args.get("expression", "")
                calc_res = execute_calculate(expr)
                if verbose:
                    print(f"      🧮 Calculation: '{expr}' = {calc_res}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": calc_res
                })
        else:
            ans = msg.content or ""
            if verbose:
                preview = (ans[:140] + "...") if len(ans) > 140 else ans
                print(f"      ✅ Math Agent Result: {preview.replace(chr(10), ' ')}")
            return ans

    # Safe text summary fallback
    try:
        flat_messages = flatten_messages_for_text_summary(
            messages,
            "State the final computed calculation results clearly."
        )
        resp_calc = groq_client.chat.completions.create(
            model=MATH_AGENT_MODEL,
            messages=flat_messages
        )
        tracker.add_usage(resp_calc.usage, step_name="Math Agent Summary")
        return resp_calc.choices[0].message.content or "Calculation completed."
    except Exception:
        tool_outputs = [m["content"] for m in messages if isinstance(m, dict) and m.get("role") == "tool"]
        return ", ".join(tool_outputs) if tool_outputs else "Calculation done."


# ============================================================
# MANAGER AGENT (ORCHESTRATOR)
# Model: MANAGER_MODEL (openai/gpt-oss-120b)
# ============================================================
MANAGER_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "delegate_to_search_agent",
            "description": "Delegates a factual/data retrieval sub-task to the specialized Web Search Agent. Parameter 'instruction' is required.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "Specific research instructions and queries for the search agent"
                    }
                },
                "required": ["instruction"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delegate_to_math_agent",
            "description": "Delegates a computation sub-task to the specialized Math Agent. Parameter 'instruction' is required.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "Specific arithmetic tasks or expressions with raw numbers to compute"
                    }
                },
                "required": ["instruction"]
            }
        }
    }
]


def run_multi_agent(user_query: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Executes a Multi-Agent architecture:
    1. Manager Agent (openai/gpt-oss-120b) decomposes and directs the task.
    2. Search Agent (openai/gpt-oss-20b) retrieves facts/numbers using Tavily.
    3. Math Agent (qwen/qwen3.8-27b) calculates arithmetic results using AST evaluator.
    
    Tracks token consumption per agent individually and in aggregate.
    """
    manager_tracker = TokenTracker(name="Manager Agent", model=MANAGER_MODEL)
    search_tracker = TokenTracker(name="Search Agent", model=SEARCH_AGENT_MODEL)
    math_tracker = TokenTracker(name="Math Agent", model=MATH_AGENT_MODEL)

    start_time = time.time()

    if verbose:
        print("\n" + "=" * 75)
        print("👥 [MULTI-AGENT SYSTEM]")
        print("   Architecture: Hierarchical Multi-Agent with 3 Distinct LLMs")
        print(f"   • Manager Agent:  {MANAGER_MODEL}")
        print(f"   • Search Agent:   {SEARCH_AGENT_MODEL}")
        print(f"   • Math Agent:     {MATH_AGENT_MODEL}")
        print(f"   Task:             {user_query}")
        print("=" * 75)

    manager_messages = [
        {
            "role": "system",
            "content": (
                "You are an executive Project Manager Agent coordinating specialized sub-agents.\n"
                "You do not execute search or calculation directly; you delegate to your specialist agents:\n"
                "1. `delegate_to_search_agent(instruction)`: Delegate search, statistics, facts, or data gathering.\n"
                "2. `delegate_to_math_agent(instruction)`: Delegate calculations, differences, ratios, percentages.\n\n"
                "Rules:\n"
                "- First, delegate data gathering to `delegate_to_search_agent`.\n"
                "- Once the numbers are obtained, delegate the calculations to `delegate_to_math_agent`.\n"
                "- After both agents have reported back, present the final synthesized report to the user."
            )
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    max_steps = 6
    final_answer = ""
    delegation_history = []

    for step in range(1, max_steps + 1):
        if verbose:
            print(f"\n--- [Manager Step {step}] Planning & Decision ---")

        try:
            response = groq_client.chat.completions.create(
                model=MANAGER_MODEL,
                messages=manager_messages,
                tools=MANAGER_TOOLS_SCHEMA,
                tool_choice="auto"
            )
            manager_tracker.add_usage(response.usage, step_name=f"Manager Step {step}")
            msg = response.choices[0].message
        except BadRequestError:
            # Model outputted final text or format quirk
            break

        tool_calls = msg.tool_calls

        if tool_calls:
            manager_messages.append(msg)
            for tool_call in tool_calls:
                func_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except Exception:
                    args = {}

                instruction = args.get("instruction", "")

                if func_name == "delegate_to_search_agent":
                    if verbose:
                        print(f"   📢 Manager -> Delegating to SEARCH AGENT ({SEARCH_AGENT_MODEL})")
                    sub_result = run_search_agent(instruction, search_tracker, verbose=verbose)
                elif func_name == "delegate_to_math_agent":
                    if verbose:
                        print(f"   📢 Manager -> Delegating to MATH AGENT ({MATH_AGENT_MODEL})")
                    sub_result = run_math_agent(instruction, math_tracker, verbose=verbose)
                else:
                    sub_result = f"Unknown delegation tool: {func_name}"

                delegation_history.append({
                    "step": step,
                    "target_agent": func_name,
                    "instruction": instruction,
                    "result": sub_result
                })

                manager_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": sub_result
                })
        else:
            final_answer = msg.content or ""
            if verbose:
                print(f"   🎯 Manager Synthesized Final Solution")
            break

    # If final answer was not formulated directly in tool loop, synthesize cleanly via flattened messages
    if not final_answer:
        if verbose:
            print("   🔄 [Manager] Synthesizing comprehensive final report...")
        flat_messages = flatten_messages_for_text_summary(
            manager_messages,
            "Synthesize all retrieved facts and math results above into a complete, professional final answer for the user."
        )
        resp_final = groq_client.chat.completions.create(
            model=MANAGER_MODEL,
            messages=flat_messages
        )
        manager_tracker.add_usage(resp_final.usage, step_name="Manager Final Synthesis")
        final_answer = resp_final.choices[0].message.content or ""

    elapsed = time.time() - start_time

    # Compute Aggregate Totals
    total_calls = manager_tracker.api_calls + search_tracker.api_calls + math_tracker.api_calls
    total_prompt = manager_tracker.prompt_tokens + search_tracker.prompt_tokens + math_tracker.prompt_tokens
    total_completion = manager_tracker.completion_tokens + search_tracker.completion_tokens + math_tracker.completion_tokens
    total_tokens = manager_tracker.total_tokens + search_tracker.total_tokens + math_tracker.total_tokens

    if verbose:
        print("\n" + "=" * 75)
        print("📋 [FINAL ANSWER - MULTI-AGENT]")
        print("=" * 75)
        print(final_answer.strip())
        print("\n" + "=" * 75)
        print("📊 [TOKEN CONSUMPTION REPORT - MULTI-AGENT SYSTEM]")
        print("=" * 75)
        
        table_rows = [
            manager_tracker.summary_row(),
            search_tracker.summary_row(),
            math_tracker.summary_row(),
            ["TOTAL (SYSTEM)", "Combined 3 Models", total_calls, total_prompt, total_completion, total_tokens]
        ]
        headers = ["Agent Role", "Model Used", "API Calls", "Prompt Tokens", "Completion Tokens", "Total Tokens"]
        print(tabulate(table_rows, headers=headers, tablefmt="fancy_grid"))
        print(f"⏱️  Total Multi-Agent Execution Time: {elapsed:.2f} seconds\n")

    return {
        "architecture": "Multi-Agent",
        "manager_tracker": manager_tracker,
        "search_tracker": search_tracker,
        "math_tracker": math_tracker,
        "totals": {
            "api_calls": total_calls,
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_tokens,
        },
        "final_answer": final_answer,
        "delegation_history": delegation_history,
        "elapsed_seconds": elapsed,
    }


if __name__ == "__main__":
    task = (
        "Search for the current population of Tokyo and the population of New York City, "
        "and calculate the difference between their populations and the percentage by which Tokyo is larger."
    )
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    run_multi_agent(task)
