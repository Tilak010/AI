from typing import TypedDict
from langgraph.graph import StateGraph, END

class State(TypedDict):
    number: int

def double(state: State):
    board_num = state['number']
    new_num = board_num * 2
    print(f"In double: {new_num}")
    return {"number": new_num}

def finish(state: State):
    print(f"In finish: {state['number']}")
    return {"number": state['number']}

def decision(state: State):
    if state['number'] < 100:
        return "double"
    else:
        return "finish"

builder = StateGraph(State)
builder.add_node("double", double)
builder.add_node("finish", finish)

builder.set_entry_point("double")

builder.add_conditional_edges(
    "double",
    decision,
    {"double": "double", "finish": "finish"}
)

builder.add_edge("finish", END)

graph = builder.compile()

if __name__ == "__main__":
    result = graph.invoke({"number": 5})
    print("Final result:", result)