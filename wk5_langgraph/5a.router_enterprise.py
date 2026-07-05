"""
5a. Enterprise Customer Service Router — CLI Demo

A LangGraph router that classifies customer messages and routes them
to specialised handler nodes (order_status, technical, billing, general).

Usage:
    python 5a.router_enterprise.py
    python 5a.router_enterprise.py "Where is my order ORD-1234?"
    python 5a.router_enterprise.py --interactive
"""

import sys
sys.path.insert(0, "..")

from langchain_common import bootstrap_notebook, create_noreason_llm, get_databricks_config

from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage

# ─── Bootstrap ───────────────────────────────────────────────────────────────

DATABRICKS_TOKEN, DATABRICKS_HOST, DATABRICKS_MODEL, (llm, llm_noreason), embeddings = bootstrap_notebook()

# ─── State ───────────────────────────────────────────────────────────────────

class CustomerServiceState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    route: str
    customer_id: str

# ─── Order Lookup Tool ───────────────────────────────────────────────────────

ORDER_DB = {
    "ORD-1234": {"status": "shipped", "eta": "July 8", "item": "Wireless Headphones"},
    "ORD-5678": {"status": "processing", "eta": "July 12", "item": "USB-C Hub"},
    "ORD-9999": {"status": "delivered", "eta": "July 1", "item": "Laptop Stand"},
}


def lookup_order(order_id: str) -> str:
    """Look up the status of a customer order by order ID.

    Args:
        order_id: The order ID to look up (e.g. ORD-1234)
    """
    order = ORDER_DB.get(order_id.upper())
    if order:
        return f"Order {order_id}: {order['item']} — Status: {order['status']}, ETA: {order['eta']}"
    return f"Order {order_id} not found. Please check the order ID."


order_tools = [lookup_order]
llm_with_order_tools = llm_noreason.bind_tools(order_tools)

# ─── Classifier Node ─────────────────────────────────────────────────────────

CATEGORIES = ["order_status", "technical", "billing", "general"]


def cs_classifier(state: CustomerServiceState):
    """Classify the customer request into a department."""
    last_msg = state["messages"][-1].content

    classification = llm_noreason.invoke([
        SystemMessage(content=(
            "You are a customer service router. Classify the customer message into exactly one category.\n"
            "Reply with ONLY the category name, nothing else.\n"
            f"Categories: {', '.join(CATEGORIES)}\n\n"
            "Guidelines:\n"
            "- order_status: tracking, shipping, delivery, order ID, where is my order\n"
            "- technical: product issues, how-to, troubleshooting, setup, compatibility\n"
            "- billing: payment, refund, invoice, charge, subscription, pricing\n"
            "- general: everything else (greetings, feedback, unrelated)"
        )),
        HumanMessage(content=last_msg),
    ])

    route = classification.content.strip().lower().replace(" ", "_")
    if route not in CATEGORIES:
        route = "general"

    print(f"  [router] → {route}")
    return {"route": route}

# ─── Handler Nodes ───────────────────────────────────────────────────────────


def order_status_node(state: CustomerServiceState):
    """Handle order status queries using the lookup tool."""
    msgs = [
        SystemMessage(content=(
            "You are an order status assistant. Use the lookup_order tool to check order status. "
            "If the customer doesn't provide an order ID, ask for it politely."
        ))
    ] + state["messages"]

    response = llm_with_order_tools.invoke(msgs)

    if response.tool_calls:
        tool_results = ToolNode(order_tools).invoke({"messages": msgs + [response]})
        final = llm_noreason.invoke(msgs + [response] + tool_results["messages"])
        return {"messages": [response] + tool_results["messages"] + [final]}

    return {"messages": [response]}


def technical_node(state: CustomerServiceState):
    """Handle technical support questions."""
    response = llm_noreason.invoke(
        [SystemMessage(content=(
            "You are a technical support specialist. Help the customer with product issues, "
            "troubleshooting, and setup questions. Be concise and helpful."
        ))] + state["messages"]
    )
    return {"messages": [response]}


def billing_node(state: CustomerServiceState):
    """Handle billing and payment questions."""
    response = llm_noreason.invoke(
        [SystemMessage(content=(
            "You are a billing support specialist. Help the customer with payment, refund, "
            "and invoice questions. Be concise. For refunds, always provide a reference number."
        ))] + state["messages"]
    )
    return {"messages": [response]}


def cs_general_node(state: CustomerServiceState):
    """Handle general customer queries."""
    response = llm_noreason.invoke(
        [SystemMessage(content=(
            "You are a friendly customer service representative. Keep responses brief and helpful."
        ))] + state["messages"]
    )
    return {"messages": [response]}

# ─── Build Graph ─────────────────────────────────────────────────────────────


def cs_route(state: CustomerServiceState):
    return state["route"]


def build_graph():
    builder = StateGraph(CustomerServiceState)

    builder.add_node("classifier", cs_classifier)
    builder.add_node("order_status", order_status_node)
    builder.add_node("technical", technical_node)
    builder.add_node("billing", billing_node)
    builder.add_node("general", cs_general_node)

    builder.add_edge(START, "classifier")
    builder.add_conditional_edges("classifier", cs_route, CATEGORIES)
    builder.add_edge("order_status", END)
    builder.add_edge("technical", END)
    builder.add_edge("billing", END)
    builder.add_edge("general", END)

    return builder.compile()

# ─── CLI ─────────────────────────────────────────────────────────────────────


def run_query(graph, message: str, customer_id: str = "CLI-USER"):
    """Run a single query through the customer service router."""
    print(f"\n{'='*60}")
    print(f"  Customer: {message}")
    print(f"{'='*60}")

    result = graph.invoke({
        "messages": [HumanMessage(content=message)],
        "customer_id": customer_id,
    })

    response = result["messages"][-1].content
    print(f"\n  Agent: {response}\n")
    return response


def interactive_mode(graph):
    """Run an interactive loop for CLI demo."""
    print("\n" + "=" * 60)
    print("  Customer Service Router — Interactive Mode")
    print("  Type 'quit' or 'exit' to stop.")
    print("  Try: 'Where is ORD-1234?', 'I need a refund', etc.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            print("  Goodbye!")
            break

        run_query(graph, user_input)


if __name__ == "__main__":
    graph = build_graph()

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode(graph)
    elif len(sys.argv) > 1:
        # Single query from command line argument
        run_query(graph, " ".join(sys.argv[1:]))
    else:
        # Default: run example queries
        examples = [
            "Where is my order ORD-1234?",
            "My headphones won't pair with my laptop via Bluetooth.",
            "I was charged twice for my last order. Can I get a refund?",
            "What are your store hours?",
        ]
        for msg in examples:
            run_query(graph, msg)
