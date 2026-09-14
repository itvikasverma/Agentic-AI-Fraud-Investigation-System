import operator
from typing import Annotated, Sequence, TypedDict, List
from langchain_core.messages import BaseMessage

class AgentTraceStep(TypedDict):
    agent_name: str
    action: str
    timestamp: str
    status: str
    short_result: str

class InvestigationState(TypedDict):
    # Inputs
    transaction_id: str
    customer_id: str
    transaction_data: dict
    
    # ML Results
    fraud_probability: float
    anomaly_score: float
    ml_explanation: list
    
    # Evidence Gathering
    sql_evidence: str
    rag_evidence: str
    
    # Analysis
    risk_level: str
    investigation_summary: str
    recommended_action: str
    
    # Human-in-the-Loop
    requires_human_review: bool
    human_decision: str
    
    # System
    messages: Annotated[Sequence[BaseMessage], operator.add]
    agent_trace: Annotated[List[AgentTraceStep], operator.add]
    errors: list
    next_agent: str
