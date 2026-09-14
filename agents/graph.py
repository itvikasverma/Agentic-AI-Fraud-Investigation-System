import os
import json
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from agents.state import InvestigationState, AgentTraceStep
from tools.ml_tool import get_ml_fraud_assessment
from tools.sql_tool import execute_read_only_sql
from tools.rag_tool import search_fraud_knowledge_base

def get_llm():
    model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    return ChatGroq(model_name=model, temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"))

def add_trace(state: InvestigationState, agent_name: str, action: str, result: str):
    trace = {
        "agent_name": agent_name,
        "action": action,
        "timestamp": datetime.utcnow().isoformat(),
        "status": "SUCCESS",
        "short_result": result[:200] + "..." if len(result) > 200 else result
    }
    return [trace]

# --- NODES ---

def ml_agent_node(state: InvestigationState):
    """Evaluates transaction with ML models."""
    tx_data = state["transaction_data"]
    result_json = get_ml_fraud_assessment.invoke(json.dumps(tx_data))
    
    try:
        result = json.loads(result_json)
        if "error" in result:
            return {"errors": [result["error"]]}
            
        prediction = result["prediction"]
        explanation = result["explanation"]
        
        trace = add_trace(
            state, "Fraud ML Agent", "Evaluate Transaction", 
            f"Prob: {prediction['fraud_probability']:.2f}, Risk: {prediction['risk_level']}"
        )
        
        return {
            "fraud_probability": prediction["fraud_probability"],
            "anomaly_score": prediction["anomaly_score"],
            "risk_level": prediction["risk_level"], # Initial risk level
            "ml_explanation": explanation["top_features"],
            "agent_trace": trace
        }
    except Exception as e:
        return {"errors": [str(e)]}

def supervisor_agent_node(state: InvestigationState):
    """Decides if further investigation (SQL/RAG) is needed based on ML results."""
    prob = state.get("fraud_probability", 0.0)
    
    if prob < 0.30:
        # Low risk, no deep investigation needed
        trace = add_trace(state, "Supervisor Agent", "Route Workflow", "Routed to Investigation Agent (Low Risk)")
        return {"next_agent": "investigate", "agent_trace": trace}
    else:
        # Medium to High risk, gather evidence
        trace = add_trace(state, "Supervisor Agent", "Route Workflow", "Routed to Evidence Gathering (Medium/High Risk)")
        return {"next_agent": "gather_evidence", "agent_trace": trace}

def sql_agent_node(state: InvestigationState):
    """Retrieves customer history from Postgres."""
    llm = get_llm()
    prompt = f"""
    You are a SQL Agent for a Fraud Investigation System.
    Write a SQLite query to retrieve the customer's transaction history and averages.
    Customer ID: {state['customer_id']}
    Tables and their exactly named columns:
    - customers: customer_id, name, account_age_days, average_transaction, transaction_frequency_per_month, usual_locations, usual_devices, previous_fraud_count
    - transactions: transaction_id, customer_id, amount, timestamp, location, device_id, merchant_category, transaction_hour, weekend, location_change, device_change, international_transaction, failed_attempts, distance_from_previous_transaction, transaction_velocity, amount_deviation, status, fraud_label
    Only output the raw SQL query. Do not use markdown formatting or explain.
    """
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        query = response.content.replace("```sql", "").replace("```", "").strip()
        
        evidence = execute_read_only_sql.invoke(query)
        trace = add_trace(state, "SQL Agent", f"Query Customer History ({query})", evidence)
        
        return {"sql_evidence": evidence, "agent_trace": trace}
    except Exception as e:
        return {"errors": [f"SQL Agent Error: {str(e)}"]}

def rag_agent_node(state: InvestigationState):
    """Retrieves policies and historical cases from Qdrant."""
    llm = get_llm()
    # Create a summary of the transaction for semantic search
    tx_summary = f"Transaction amount {state['transaction_data'].get('amount', 0)}. "
    if state['transaction_data'].get('location_change'):
        tx_summary += "Location changed. "
    if state['transaction_data'].get('device_change'):
        tx_summary += "New device used. "
        
    top_features = [f["feature"] for f in state.get("ml_explanation", [])[:3]]
    tx_summary += f"Key risk factors: {', '.join(top_features)}."
    
    evidence = search_fraud_knowledge_base.invoke(tx_summary)
    trace = add_trace(state, "RAG Agent", f"Search Knowledge Base ({tx_summary})", evidence)
    
    return {"rag_evidence": evidence, "agent_trace": trace}

def investigation_agent_node(state: InvestigationState):
    """Synthesizes all evidence into a final report."""
    llm = get_llm()
    prompt = f"""
    You are a Lead Fraud Investigator. Synthesize the following evidence into a concise investigation report.
    Do NOT invent information.
    
    Transaction Data: {json.dumps(state['transaction_data'])}
    Fraud Probability: {state.get('fraud_probability')}
    Anomaly Score: {state.get('anomaly_score')}
    ML Explanation (SHAP): {json.dumps(state.get('ml_explanation', []))}
    SQL Evidence (Customer History): {state.get('sql_evidence', 'N/A')}
    RAG Evidence (Policies & Cases): {state.get('rag_evidence', 'N/A')}
    
    Provide:
    1. Summary
    2. Key Risk Factors
    3. Policy Violations (if any)
    
    IMPORTANT: Format all monetary amounts strictly in Indian Rupees (₹ or INR). Do NOT use USD ($).
    """
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        report = response.content
        trace = add_trace(state, "Investigation Agent", "Synthesize Evidence", "Report generated successfully.")
        
        return {"investigation_summary": report, "agent_trace": trace}
    except Exception as e:
        return {"errors": [f"Investigator Error: {str(e)}"]}

def risk_agent_node(state: InvestigationState):
    """Determines final risk, recommendation, and HITL requirement."""
    prob = state.get("fraud_probability", 0.0)
    
    # Simple rule-based risk assignment based on ML, can be enhanced by LLM
    risk_level = "LOW"
    action = "APPROVE"
    hitl = False
    
    if prob > 0.90:
        risk_level = "CRITICAL"
        action = "BLOCK"
        hitl = True
    elif prob > 0.70:
        risk_level = "HIGH"
        action = "REQUEST VERIFICATION"
        hitl = True
    elif prob > 0.30:
        risk_level = "MEDIUM"
        action = "REVIEW"
    
    trace = add_trace(state, "Risk Agent", "Determine Action", f"Risk: {risk_level}, Action: {action}, HITL: {hitl}")
    
    return {
        "risk_level": risk_level,
        "recommended_action": action,
        "requires_human_review": hitl,
        "agent_trace": trace
    }

# --- GRAPH DEFINITION ---

def build_graph():
    workflow = StateGraph(InvestigationState)
    
    workflow.add_node("ml_agent", ml_agent_node)
    workflow.add_node("supervisor", supervisor_agent_node)
    workflow.add_node("sql_agent", sql_agent_node)
    workflow.add_node("rag_agent", rag_agent_node)
    workflow.add_node("investigator", investigation_agent_node)
    workflow.add_node("risk_assessor", risk_agent_node)
    
    workflow.set_entry_point("ml_agent")
    workflow.add_edge("ml_agent", "supervisor")
    
    # Conditional routing
    workflow.add_conditional_edges(
        "supervisor",
        lambda x: x["next_agent"],
        {
            "gather_evidence": "sql_agent",
            "investigate": "investigator"
        }
    )
    
    # Parallel evidence gathering
    workflow.add_edge("sql_agent", "rag_agent")
    workflow.add_edge("rag_agent", "investigator")
    
    workflow.add_edge("investigator", "risk_assessor")
    workflow.add_edge("risk_assessor", END)
    
    return workflow.compile()
