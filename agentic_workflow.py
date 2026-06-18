import operator
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from datetime import datetime
from langgraph.graph import StateGraph, END

# ---------------------------------------------------------
# 3.3 WORKFLOW STATE STORE & 3.5 CONTEXT & MEMORY
# ---------------------------------------------------------
class ExceptionData(TypedDict):
    exception_id: str
    break_type: str
    discrepancy_usd: float
    status: str

class WorkflowStateStore(TypedDict):
    # Core Exception Data
    exception: ExceptionData
    
    # 3.3 Workflow State Store
    status: str
    owner: str
    next_skill: str
    evidence: Annotated[List[Dict[str, Any]], operator.add]
    approvals: List[str]
    audit_trail: Annotated[List[str], operator.add]
    timestamps: Dict[str, str]
    
    # 3.5 Context & Memory
    exception_history: List[Dict]
    fund_context: Dict
    market_context: Dict
    policies_and_rules: Dict
    user_context: Dict
    regulatory_context: Dict
    
    # Execution
    messages: Annotated[List[Any], operator.add]
    severity: str
    priority_score: float
    investigation_plan: List[str]
    triage_summary: str

# ---------------------------------------------------------
# MCP SERVERS (Tool Endpoints Called by Skills)
# ---------------------------------------------------------
def bloomberg_mcp(query: str):
    return {"source": "Bloomberg MCP", "data": "Pricing data..."}

def factset_mcp(query: str):
    return {"source": "FactSet MCP", "data": "FactSet data..."}

def custodian_mcp(query: str):
    return {"source": "Custodian MCP", "data": "Custodian records..."}

# ---------------------------------------------------------
# 3.1 EXCEPTION INTELLIGENCE
# ---------------------------------------------------------
def exception_intelligence_node(state: WorkflowStateStore):
    """
    Determine Severity, Classify Exception Type, Prioritize & Score,
    Create Investigation Plan, Initial Triage Summary.
    """
    logs = [f"[{datetime.now()}] Exception Intelligence Phase Started"]
    
    discrepancy = state['exception'].get('discrepancy_usd', 0)
    
    # Determine Severity & Score
    if discrepancy > 50000:
        severity = "High"
        priority_score = 9.5
    elif discrepancy > 10000:
        severity = "Medium"
        priority_score = 5.0
    else:
        severity = "Low"
        priority_score = 2.0
        
    triage_summary = f"Exception classified as {severity} priority with a score of {priority_score}."
    investigation_plan = ["Check Custodian Records", "Verify Market Pricing", "Reconcile Cash flow"]
    status = "Triaged"
    logs.append(f"[{datetime.now()}] Exception Intelligence Completed. Severity: {severity}")
    
    return {
        "severity": severity,
        "priority_score": priority_score,
        "triage_summary": triage_summary,
        "investigation_plan": investigation_plan,
        "status": status,
        "audit_trail": logs
    }

# ---------------------------------------------------------
# RECONCILIATION SKILL (Called by Supervisor)
# ---------------------------------------------------------
def reconciliation_skill_node(state: WorkflowStateStore):
    """
    Handles Cash Breaks, Position Breaks, Trade Breaks, Cash FX / Custodian.
    """
    logs = [f"[{datetime.now()}] Reconciliation Skill Started"]
    break_type = state['exception'].get('break_type')
    
    new_evidence = []
    
    # Call MCP Tools based on break type
    if break_type == "Position Break":
        new_evidence.append(bloomberg_mcp("Get position pricing"))
    elif break_type == "Cash Break":
        new_evidence.append(custodian_mcp("Get cash statement"))
        
    logs.append(f"[{datetime.now()}] Reconciliation Skill Completed using MCPs.")
    
    return {
        "evidence": new_evidence,
        "status": "Reconciled",
        "audit_trail": logs
    }

# ---------------------------------------------------------
# 3.2 LANGGRAPH ORCHESTRATOR (SUPERVISOR)
# ---------------------------------------------------------
def supervisor_node(state: WorkflowStateStore):
    """
    Dynamic Skill Routing, Escalation Logic, Workflow Orchestration.
    """
    logs = [f"[{datetime.now()}] Supervisor evaluating next steps..."]
    
    next_skill = state.get('next_skill')
    
    if state.get('status') == "Triaged":
        # Route to Reconciliation Skill
        next_skill = "reconciliation_skill"
    elif state.get('status') == "Reconciled":
        if state.get('severity') == "High":
            next_skill = "escalation"
        else:
            next_skill = "complete"
            
    return {
        "next_skill": next_skill,
        "audit_trail": logs
    }

def supervisor_router(state: WorkflowStateStore):
    next_skill = state.get('next_skill')
    if next_skill == "reconciliation_skill":
        return "reconciliation_skill_node"
    elif next_skill == "escalation":
        return "escalation_node"
    elif next_skill == "complete":
        return "complete_node"
    return END

def escalation_node(state: WorkflowStateStore):
    logs = [f"[{datetime.now()}] Escalated for Manual Review."]
    return {
        "status": "Escalated",
        "audit_trail": logs
    }

def complete_node(state: WorkflowStateStore):
    logs = [f"[{datetime.now()}] Workflow Completed."]
    return {
        "status": "Completed",
        "audit_trail": logs
    }

# ---------------------------------------------------------
# BUILD THE GRAPH
# ---------------------------------------------------------
def build_agentic_workflow():
    workflow = StateGraph(WorkflowStateStore)
    
    # Add Nodes
    workflow.add_node("exception_intelligence", exception_intelligence_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("reconciliation_skill_node", reconciliation_skill_node)
    workflow.add_node("escalation_node", escalation_node)
    workflow.add_node("complete_node", complete_node)
    
    # Define Edges (Orchestration)
    workflow.set_entry_point("exception_intelligence")
    
    # After Intelligence -> Supervisor
    workflow.add_edge("exception_intelligence", "supervisor")
    
    # Supervisor dynamic routing
    workflow.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "reconciliation_skill_node": "reconciliation_skill_node",
            "escalation_node": "escalation_node",
            "complete_node": "complete_node",
            END: END
        }
    )
    
    # After a skill, always return to Supervisor
    workflow.add_edge("reconciliation_skill_node", "supervisor")
    workflow.add_edge("escalation_node", END)
    workflow.add_edge("complete_node", END)
    
    return workflow.compile()

if __name__ == "__main__":
    app = build_agentic_workflow()
    
    # Mock input
    initial_state = {
        "exception": {
            "exception_id": "EX-1001",
            "break_type": "Position Break",
            "discrepancy_usd": 65000.0,
            "status": "New"
        },
        "audit_trail": [],
        "evidence": [],
    }
    
    print("--- Starting Agentic Workflow ---")
    final_state = app.invoke(initial_state)
    
    print("\n--- Final State Summary ---")
    print(f"Status: {final_state.get('status')}")
    print(f"Severity: {final_state.get('severity')}")
    print(f"Priority Score: {final_state.get('priority_score')}")
    print(f"Evidence Collected: {final_state.get('evidence')}")
    
    print("\n--- Audit Trail ---")
    for log in final_state.get('audit_trail', []):
        print(log)
