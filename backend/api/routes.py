from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import json
import redis
import os

from backend.schemas.api_models import TransactionRequest, DecisionRequest, InvestigationResponse
from database.connection import get_db
from database.models import Transaction, InvestigationLog, Decision
from agents.graph import build_graph

router = APIRouter()
graph = build_graph()

# Setup Redis
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", 6379))
try:
    redis_client = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)
    redis_client.ping()
except Exception:
    print("Warning: Could not connect to Redis. Caching disabled.")
    redis_client = None

@router.post("/api/investigate", response_model=InvestigationResponse)
async def investigate_transaction(request: TransactionRequest, db: Session = Depends(get_db)):
    tx_id = request.transaction_id
    
    # Check cache
    if redis_client:
        cached_result = redis_client.get(f"investigation:{tx_id}")
        if cached_result:
            return json.loads(cached_result)

    # Initialize state
    initial_state = {
        "transaction_id": tx_id,
        "customer_id": request.customer_id,
        "transaction_data": request.model_dump(),
        "agent_trace": [],
        "errors": []
    }
    
    # Run LangGraph
    try:
        final_state = graph.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    # Save investigation log to DB
    log = InvestigationLog(
        transaction_id=tx_id,
        fraud_probability=final_state.get("fraud_probability"),
        anomaly_score=final_state.get("anomaly_score"),
        risk_level=final_state.get("risk_level"),
        ml_explanation=json.dumps(final_state.get("ml_explanation", [])),
        sql_evidence=final_state.get("sql_evidence"),
        rag_evidence=final_state.get("rag_evidence"),
        summary=final_state.get("investigation_summary"),
        recommended_action=final_state.get("recommended_action")
    )
    db.add(log)
    db.commit()
    
    response = InvestigationResponse(
        transaction_id=tx_id,
        fraud_probability=final_state.get("fraud_probability", 0.0),
        anomaly_score=final_state.get("anomaly_score", 0.0),
        risk_level=final_state.get("risk_level", "UNKNOWN"),
        recommended_action=final_state.get("recommended_action", "REVIEW"),
        requires_human_review=final_state.get("requires_human_review", False),
        ml_explanation=final_state.get("ml_explanation", []),
        investigation_summary=final_state.get("investigation_summary"),
        agent_trace=final_state.get("agent_trace", []),
        errors=final_state.get("errors", [])
    )
    
    # Cache result (TTL 1 hour)
    if redis_client:
        redis_client.setex(f"investigation:{tx_id}", 3600, response.model_dump_json())
        
    return response

@router.post("/api/decision")
async def submit_decision(request: DecisionRequest, db: Session = Depends(get_db)):
    # Find investigation log
    log = db.query(InvestigationLog).filter(InvestigationLog.transaction_id == request.transaction_id).order_by(InvestigationLog.created_at.desc()).first()
    
    if not log:
        raise HTTPException(status_code=404, detail="Investigation log not found")
        
    decision = Decision(
        log_id=log.log_id,
        human_decision=request.decision,
        decided_by=request.decided_by,
        notes=request.notes
    )
    db.add(decision)
    
    # Update transaction status
    tx = db.query(Transaction).filter(Transaction.transaction_id == request.transaction_id).first()
    if tx:
        tx.status = request.decision
        
    db.commit()
    return {"status": "success"}

@router.get("/api/health")
async def health_check():
    return {"status": "healthy"}

@router.get("/api/history")
async def get_history(db: Session = Depends(get_db)):
    # Get latest decisions with their corresponding investigation logs
    decisions = db.query(Decision).order_by(Decision.created_at.desc()).limit(50).all()
    history = []
    for d in decisions:
        log = db.query(InvestigationLog).filter(InvestigationLog.log_id == d.log_id).first()
        if log:
            history.append({
                "transaction_id": log.transaction_id,
                "fraud_probability": log.fraud_probability,
                "risk_level": log.risk_level,
                "ai_recommendation": log.recommended_action,
                "human_decision": d.human_decision,
                "decided_by": d.decided_by,
                "timestamp": d.created_at.isoformat()
            })
    return history
