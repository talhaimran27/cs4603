"""
11a. Databricks Deployment Setup Script

Run this script in a Databricks notebook or from a terminal with
the Databricks CLI configured. It creates all the prerequisites
needed for the deployment notebook (11.deployment.ipynb):

  1. Unity Catalog schema for the model
  2. MLflow experiment
  3. Logs the LangGraph agent as an MLflow model
  4. Registers the model in Unity Catalog
  5. Creates a Model Serving endpoint

Prerequisites:
  - Databricks CLI configured (`databricks auth login`) OR
  - Running inside a Databricks notebook with workspace auth
  - .env file with DATABRICKS_TOKEN, DATABRICKS_HOST, DATABRICKS_MODEL

Usage (from repo root):
    python wk5_langgraph/11a.deploy_setup.py

    # Or with custom catalog/schema:
    python wk5_langgraph/11a.deploy_setup.py --catalog my_catalog --schema my_schema

    # Skip endpoint creation (just register model):
    python wk5_langgraph/11a.deploy_setup.py --skip-endpoint
"""

import argparse
import sys
import time

sys.path.insert(0, "..")

# ─── Parse arguments ─────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Set up Databricks prerequisites for LangGraph deployment")
parser.add_argument("--catalog", default="cs4603", help="Unity Catalog name (default: cs4603)")
parser.add_argument("--schema", default="cs4603-langgraph", help="Schema name (default: cs4603-langgraph)")
parser.add_argument("--model-name", default="cs4603-langgraph_agent", help="Registered model name (default: cs4603-langgraph_agent)")
parser.add_argument("--endpoint-name", default="cs4603-langgraph-agent", help="Serving endpoint name (default: cs4603-langgraph-agent)")
parser.add_argument("--skip-endpoint", action="store_true", help="Skip creating the serving endpoint")
args = parser.parse_args()

UC_MODEL_PATH = f"{args.catalog}.{args.schema}.{args.model_name}"

# ─── Bootstrap ───────────────────────────────────────────────────────────────

print("=" * 60)
print("  LangGraph Agent — Databricks Deployment Setup")
print("=" * 60)

from langchain_common import bootstrap_notebook

DATABRICKS_TOKEN, DATABRICKS_HOST, DATABRICKS_MODEL, (llm, llm_noreason), embeddings = bootstrap_notebook()
print(f"\n  Databricks host: {DATABRICKS_HOST}")
print(f"  Model endpoint:  {DATABRICKS_MODEL}")

# ─── Step 1: Create Unity Catalog schema ─────────────────────────────────────

print(f"\n{'─'*60}")
print(f"  Step 1: Ensure Unity Catalog schema exists")
print(f"  Path: {args.catalog}.{args.schema}")
print(f"{'─'*60}")

try:
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient(
        host=DATABRICKS_HOST,
        token=DATABRICKS_TOKEN,
    )

    # Create catalog if it doesn't exist
    try:
        w.catalogs.get(args.catalog)
        print(f"  ✓ Catalog '{args.catalog}' exists")
    except Exception:
        print(f"  Creating catalog '{args.catalog}'...")
        w.catalogs.create(name=args.catalog)
        print(f"  ✓ Catalog '{args.catalog}' created")

    # Create schema if it doesn't exist
    try:
        w.schemas.get(f"{args.catalog}.{args.schema}")
        print(f"  ✓ Schema '{args.catalog}.{args.schema}' exists")
    except Exception:
        print(f"  Creating schema '{args.catalog}.{args.schema}'...")
        w.schemas.create(name=args.schema, catalog_name=args.catalog)
        print(f"  ✓ Schema '{args.catalog}.{args.schema}' created")

    HAS_SDK = True

except ImportError:
    print("  ⚠ databricks-sdk not installed — skipping catalog/schema creation.")
    print("    Install with: pip install databricks-sdk")
    print("    You'll need to create the catalog/schema manually in the Databricks UI.")
    HAS_SDK = False

except Exception as e:
    print(f"  ✗ Could not create catalog/schema: {e}")
    print("    Ensure Unity Catalog is enabled and a metastore is assigned to this workspace.")
    sys.exit(1)

# ─── Step 2: Define the LangGraph agent ──────────────────────────────────────

print(f"\n{'─'*60}")
print(f"  Step 2: Define the LangGraph agent")
print(f"{'─'*60}")

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage


def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b


def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


tools = [multiply, add]
llm_with_tools = llm.bind_tools(tools)


def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def route_tools(state: MessagesState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", route_tools, ["tools", END])
builder.add_edge("tools", "assistant")

graph = builder.compile()

# Sanity check
result = graph.invoke({"messages": [HumanMessage(content="Multiply 3 by 2.")]})
answer = result["messages"][-1].content
print(f"  ✓ Agent compiled and tested (3×2 = {answer})")

# ─── Step 3: Log to MLflow ───────────────────────────────────────────────────

print(f"\n{'─'*60}")
print(f"  Step 3: Log agent to MLflow")
print(f"{'─'*60}")

import mlflow
import os

# Point MLflow at the Databricks workspace from .env (not local sqlite)
os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

print(f"  MLflow tracking: {mlflow.get_tracking_uri()}")
print(f"  MLflow registry: {mlflow.get_registry_uri()}")
print(f"  Target host:     {DATABRICKS_HOST}")

mlflow.set_experiment("/cs4603/wk5-deployment")

with mlflow.start_run(run_name="langgraph-agent-setup") as run:
    model_info = mlflow.langchain.log_model(
        lc_model=graph,
        artifact_path="langgraph_agent",
        input_example={"messages": [{"role": "user", "content": "Add 2 and 3."}]},
    )
    run_id = run.info.run_id

print(f"  ✓ Model logged: {model_info.model_uri}")
print(f"  ✓ Run ID: {run_id}")

# ─── Step 4: Register in Unity Catalog ────────────────────────────────────────

print(f"\n{'─'*60}")
print(f"  Step 4: Register model in Unity Catalog")
print(f"  Path: {UC_MODEL_PATH}")
print(f"{'─'*60}")

try:
    mlflow.set_registry_uri("databricks-uc")

    registered = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=UC_MODEL_PATH,
    )
    model_version = registered.version
    print(f"  ✓ Registered '{UC_MODEL_PATH}' version {model_version}")

except Exception as e:
    print(f"  ⚠ Could not register in Unity Catalog: {e}")
    print("    Falling back to workspace registry...")

    try:
        from mlflow import MlflowClient

        client = MlflowClient()
        try:
            client.create_registered_model(args.model_name)
        except Exception:
            pass  # Already exists

        mv = client.create_model_version(
            name=args.model_name,
            source=model_info.model_uri,
            run_id=run_id,
        )
        model_version = mv.version
        UC_MODEL_PATH = args.model_name
        print(f"  ✓ Registered '{args.model_name}' version {model_version} (workspace registry)")
    except Exception as e2:
        print(f"  ✗ Registration failed: {e2}")
        model_version = "1"

# ─── Step 5: Create serving endpoint ─────────────────────────────────────────

if args.skip_endpoint:
    print(f"\n{'─'*60}")
    print(f"  Step 5: Skipped (--skip-endpoint)")
    print(f"{'─'*60}")
else:
    print(f"\n{'─'*60}")
    print(f"  Step 5: Create Model Serving endpoint")
    print(f"  Endpoint: {args.endpoint_name}")
    print(f"{'─'*60}")

    if not HAS_SDK:
        print("  ⚠ databricks-sdk not available — cannot create endpoint automatically.")
        print(f"    Create it manually in the Databricks UI:")
        print(f"    - Go to Serving → New → select '{UC_MODEL_PATH}' version {model_version}")
        print(f"    - Name it '{args.endpoint_name}'")
        print(f"    - Enable 'Scale to zero'")
    else:
        try:
            from databricks.sdk.service.serving import (
                EndpointCoreConfigInput,
                ServedEntityInput,
            )

            # Check if endpoint already exists
            existing = None
            try:
                existing = w.serving_endpoints.get(args.endpoint_name)
            except Exception:
                pass

            if existing:
                print(f"  Endpoint '{args.endpoint_name}' already exists (state: {existing.state.ready})")
                print(f"  Updating to model version {model_version}...")
                w.serving_endpoints.update_config(
                    name=args.endpoint_name,
                    served_entities=[
                        ServedEntityInput(
                            entity_name=UC_MODEL_PATH,
                            entity_version=str(model_version),
                            workload_size="Small",
                            scale_to_zero_enabled=True,
                        )
                    ],
                )
                print(f"  ✓ Endpoint updated")
            else:
                print(f"  Creating endpoint '{args.endpoint_name}'...")
                w.serving_endpoints.create(
                    name=args.endpoint_name,
                    config=EndpointCoreConfigInput(
                        served_entities=[
                            ServedEntityInput(
                                entity_name=UC_MODEL_PATH,
                                entity_version=str(model_version),
                                workload_size="Small",
                                scale_to_zero_enabled=True,
                            )
                        ]
                    ),
                )
                print(f"  ✓ Endpoint '{args.endpoint_name}' created")
                print(f"    It may take a few minutes to become READY.")

            print(f"\n  Endpoint URL: {DATABRICKS_HOST}/serving-endpoints/{args.endpoint_name}/invocations")

        except Exception as e:
            print(f"  ⚠ Could not create endpoint: {e}")
            print(f"    Create it manually in the Databricks UI.")

# ─── Summary ─────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"  Setup Complete!")
print(f"{'='*60}")
print(f"""
  Model:     {UC_MODEL_PATH} (version {model_version})
  Endpoint:  {args.endpoint_name}
  Run ID:    {run_id}

  To test the endpoint (once READY):

    import openai
    client = openai.OpenAI(
        api_key="<your-token>",
        base_url="{DATABRICKS_HOST}/serving-endpoints",
    )
    resp = client.chat.completions.create(
        model="{args.endpoint_name}",
        messages=[{{"role": "user", "content": "Multiply 3 by 2."}}],
    )
    print(resp.choices[0].message.content)
""")
