from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class TransactionRequest(BaseModel):
    transaction_id: str
    customer_id: str
    amount: float
    location: str
    device_id: str
    merchant_category: str
    transaction_hour: int
    weekend: bool
    location_change: bool
    device_change: bool
    international_transaction: bool
    failed_attempts: int
    distance_from_previous_transaction: float
    transaction_velocity: float
    amount_deviation: float

class DecisionRequest(BaseModel):
    transaction_id: str
    decision: str
    decided_by: str = "human_reviewer"
    notes: Optional[str] = None
    
class AgentTraceResponse(BaseModel):
    agent_name: str
    action: str
    timestamp: str
    status: str
    short_result: str

class InvestigationResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    anomaly_score: float
    risk_level: str
    recommended_action: str
    requires_human_review: bool
    ml_explanation: List[Dict[str, Any]]
    investigation_summary: Optional[str]
    agent_trace: List[AgentTraceResponse]
    errors: Optional[List[str]] = []
