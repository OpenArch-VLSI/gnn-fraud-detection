import os
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated

# Adjust path so we can run from anywhere
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.models import FraudCamouflageGNN

# Define Graph State
class AgentState(TypedDict):
    node_id: int
    context: str
    report: str

def load_graph_and_model():
    print("Loading graph data...")
    device = torch.device('cpu')
    data_path = os.path.join('data', 'processed', 'graph.pt')
    data = torch.load(data_path, weights_only=False, map_location=device)
    
    print("Loading model...")
    model = FraudCamouflageGNN(data.x.size(1), 64, heads=4, dropout=0.3).to(device)
    # Run name matches configs/camo_gat.yaml. train.py saves a checkpoint
    # dict ({'model': ..., 'optimizer': ..., ...}), so unwrap the 'model' key
    # (same fix as evaluate.py).
    model_path = os.path.join('experiments', 'camo_gat_baseline', 'best_model.pt')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    return data, model

# Extract context tool logic
def extract_node_context(state: AgentState):
    node_id = state['node_id']
    data, model = load_graph_and_model()
    
    # 1. Find the 1-hop neighbors of the target node
    src = data.edge_index[0]
    dst = data.edge_index[1]
    
    # Find all edges where target node is the destination
    neighbor_mask = (dst == node_id)
    neighbor_indices = src[neighbor_mask]
    
    if len(neighbor_indices) == 0:
        return {"context": f"Node {node_id} has no incoming neighbors. It is isolated."}
        
    # 2. Run a forward pass to get attention weights
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        
    # Read the saved attention weights from the first layer
    # _alpha shape is [num_edges, heads, 1]
    attention_weights = model.conv1._alpha.squeeze(-1).mean(dim=1) # Average across heads
    
    # Extract the weights specifically for the edges connecting to our node
    node_attentions = attention_weights[neighbor_mask]
    
    # 3. Format the context for the LLM
    context_lines = [f"### Context for Node ID: {node_id}"]
    context_lines.append(f"Number of connected neighbors: {len(neighbor_indices)}")
    context_lines.append("\nAttention Weights assigned by CamouflageGAT (Higher = Model paid more attention to this neighbor):")
    
    # Sort neighbors by attention weight (highest first)
    sorted_indices = torch.argsort(node_attentions, descending=True)
    for i in sorted_indices:
        n_id = neighbor_indices[i].item()
        att = node_attentions[i].item()
        # Is this neighbor a known fraudster in the training set?
        is_fraud = "Fraud" if data.y[n_id].item() == 1 else "Legitimate"
        is_train = "Train Set" if data.train_mask[n_id].item() else "Unseen"
        
        context_lines.append(f"- Neighbor {n_id:04d} ({is_fraud}, {is_train}): Attention Score = {att:.4f}")
        
    return {"context": "\n".join(context_lines)}

def analyze_fraud(state: AgentState):
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
    
    sys_prompt = """You are an elite AI Financial Fraud Investigator.
Your job is to explain the decision-making process of a Graph Neural Network (CamouflageGAT) in simple, human-readable terms.

You will be provided with the GNN's 'Attention Weights' for a specific node and its neighbors. 
- A high attention score means the GNN strongly relied on that neighbor to make its decision.
- A low attention score means the GNN ignored that neighbor (likely because it identified the edge as a 'camouflage' edge created by a fraudster to look normal).

Write a brief 3-4 sentence professional report analyzing the node's neighborhood. Highlight which neighbors the model trusted, and which it ignored."""

    messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=state['context'])
    ]
    
    response = llm.invoke(messages)
    return {"report": response.content}

# Build LangGraph
workflow = StateGraph(AgentState)
workflow.add_node("extract_context", extract_node_context)
workflow.add_node("analyze_fraud", analyze_fraud)

workflow.add_edge(START, "extract_context")
workflow.add_edge("extract_context", "analyze_fraud")
workflow.add_edge("analyze_fraud", END)

app = workflow.compile()

def investigate_node(node_id: int):
    print(f"\n--- Initiating Investigation for Node {node_id} ---")
    final_state = app.invoke({"node_id": node_id, "context": "", "report": ""})
    print("\n--- FINAL REPORT ---")
    print(final_state["report"])
    return final_state["report"]

if __name__ == "__main__":
    # Let's investigate a random known fraudulent node
    # We will load the data just to pick one for the demo
    data_path = os.path.join('data', 'processed', 'graph.pt')
    if os.path.exists(data_path):
        data = torch.load(data_path, weights_only=False, map_location='cpu')
        # Find a fraud node in the test set
        test_fraud_nodes = torch.where((data.y == 1) & (data.test_mask == True))[0]
        if len(test_fraud_nodes) > 0:
            target_node = test_fraud_nodes[0].item()
            investigate_node(target_node)
        else:
            print("No test fraud nodes found.")
    else:
        print("Graph data not found. Run build_graph.py first.")
