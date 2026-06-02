from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class CounterState(TypedDict):
    count: int

def add(state):
    return {
        "count": state["count"]+1
    }

def even(state):
    print("Even")
    return {}

def odd(state):
    print("Odd HAHA")
    return {}

def router(state):
    if state["count"] % 2 == 0:
        return "even"
    else:
        return "odd"

build = StateGraph(CounterState)
build.add_node("add", add)
build.add_node("even", even)
build.add_node("odd", odd)

build.add_edge(START, "add")

build.add_conditional_edges("add", router, {"even": "even", "odd": "odd"})
build.add_edge("even", END)
build.add_edge("odd", END)

agent = build.compile()

result = agent.invoke({"count": 2})
print(result)