import sys
import os
import json
import ast
import operator
from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient

# Ensure UTF-8 output in Windows console environments
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing in your .env file!")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is missing in your .env file!")

# Initialize Clients
groq_client = Groq(api_key=GROQ_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# Model Selection: OpenAI open-weights model hosted on Groq
MODEL_NAME = "openai/gpt-oss-120b"


# ============================================================
# 2. DEFINE TOOL FUNCTIONS
# ============================================================

def web_search(query: str) -> str:
    """
    Search the web using Tavily Search API and return relevant results.
    """
    try:
        response = tavily_client.search(query=query, max_results=3)
        results = response.get("results", [])
        
        if not results:
            return f"No search results found for query: '{query}'"
        
        formatted_results = []
        for idx, item in enumerate(results, 1):
            title = item.get("title", "No Title")
            content = item.get("content", "No Content")
            url = item.get("url", "")
            formatted_results.append(f"[{idx}] {title}\nURL: {url}\nSummary: {content}")
        
        return "\n\n".join(formatted_results)
    except Exception as e:
        return f"Error executing web search: {str(e)}"


# Safe mathematical expression evaluator
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
    if isinstance(node, ast.Constant):  # Numbers e.g., 2, 3.14
        return node.value
    elif isinstance(node, ast.BinOp):  # Binary ops: a + b, a * b
        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)
        op_type = type(node.op)
        if op_type in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
    elif isinstance(node, ast.UnaryOp):  # Unary ops: -5, +3
        operand = _eval_ast_node(node.operand)
        op_type = type(node.op)
        if op_type in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[op_type](operand)
        raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
    else:
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def calculate(expression: str) -> str:
    """
    Evaluates a mathematical expression string (e.g. '2 * 2' or '(15 + 5) / 2') safely and returns the answer.
    """
    try:
        # Clean common math symbols
        clean_expr = expression.strip().replace("^", "**").replace("x", "*").replace("X", "*")
        parsed = ast.parse(clean_expr, mode='eval')
        result = _eval_ast_node(parsed.body)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression '{expression}': {str(e)}"


# ============================================================
# 3. TOOLS SCHEMA (Standard OpenAI / Groq Format)
# ============================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Searches the web using Tavily for up-to-date real-time information, news, events, and facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up on the web"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Performs basic mathematical calculations given a math expression string, e.g., '2 * 2' or '(15 + 5) / 2'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate (e.g., '2 * 2', '125 / 5', '45 + 120 * 2')"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# Mapping function names to actual Python functions
available_tools = {
    "web_search": web_search,
    "calculate": calculate
}


# ============================================================
# 4. AI AGENT FUNCTION
# ============================================================

def run_agent(user_query: str) -> str:
    """
    Takes any user query string, lets the Groq LLM decide which tool to call (or answer directly),
    executes the selected tool(s), and returns the final answer.
    """
    print(f"\n{'=' * 60}")
    print(f"User Query: {user_query}")
    print(f"{'=' * 60}")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant equipped with two tools:\n"
                "1. `web_search`: Use this to search the web for current facts, news, and live information. Always provide the required `query` parameter as a string.\n"
                "2. `calculate`: Use this to evaluate mathematical expressions. Always provide the required `expression` parameter as a string.\n\n"
                "Rules:\n"
                "- Call tools when needed to gather facts or compute math.\n"
                "- Only use the defined parameter names (`query` for web_search, `expression` for calculate).\n"
                "- Once you receive the tool results, provide a clear, comprehensive final answer to the user without calling further unnecessary tools."
            )
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    max_iterations = 5
    final_answer = ""

    for step in range(max_iterations):
        # Call Groq LLM with tools and tool_choice='auto'
        response = groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        # If LLM requested tool call(s)
        if tool_calls:
            # Append assistant's tool-call response to messages
            messages.append(response_message)

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                try:
                    function_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    function_args = {}

                print(f"-> [Agent Decision] Calling Tool: {function_name}")
                print(f"   Arguments: {function_args}")

                # Execute the corresponding Python function
                tool_function = available_tools.get(function_name)
                if tool_function:
                    tool_result = tool_function(**function_args)
                else:
                    tool_result = f"Error: Tool '{function_name}' not found."

                preview = str(tool_result)[:150].replace('\n', ' ')
                print(f"   Result: {preview}..." if len(str(tool_result)) > 150 else f"   Result: {tool_result}")

                # Append tool result to conversation history
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": str(tool_result)
                })
        else:
            # LLM provided final response
            final_answer = response_message.content or ""
            break
    else:
        final_answer = "Max iterations reached without final answer."

    print(f"\nFinal Answer:\n{final_answer}")
    return final_answer



# ============================================================
# 5. INTERACTIVE USER QUERY LOOP
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AI Agent Initialized (Groq LLM + Tavily Search + Math Tool)")
    print("Type your question below (or type 'exit' / 'quit' to stop).")
    print("=" * 60)
    
    while True:
        try:
            user_query = input("\nEnter your query: ").strip()
            
            if not user_query:
                continue
            
            if user_query.lower() in ["exit", "quit", "q"]:
                print("Exiting agent. Goodbye!")
                break
            
            run_agent(user_query)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting agent. Goodbye!")
            break


