from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()

class Customer(Base):
    __tablename__ = 'customers'

    customer_id = Column(String, primary_key=True)
    name = Column(String)
    account_age_days = Column(Integer)
    average_transaction = Column(Float)
    transaction_frequency_per_month = Column(Float)
    usual_locations = Column(String) # Comma-separated or JSON string
    usual_devices = Column(String) # Comma-separated or JSON string
    previous_fraud_count = Column(Integer, default=0)

    transactions = relationship("Transaction", back_populates="customer")

class Transaction(Base):
    __tablename__ = 'transactions'

    transaction_id = Column(String, primary_key=True)
    customer_id = Column(String, ForeignKey('customers.customer_id'))
    amount = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    location = Column(String)
    device_id = Column(String)
    merchant_category = Column(String)
    
    # ML Features
    transaction_hour = Column(Integer)
    weekend = Column(Boolean)
    location_change = Column(Boolean)
    device_change = Column(Boolean)
    international_transaction = Column(Boolean)
    failed_attempts = Column(Integer)
    distance_from_previous_transaction = Column(Float)
    transaction_velocity = Column(Float) # Transactions in the last hour
    amount_deviation = Column(Float) # (amount - avg_amount) / avg_amount
    
    status = Column(String, default="APPROVED") # PENDING, APPROVED, BLOCKED, REVIEW
    fraud_label = Column(Boolean, default=False) # 1 for fraud, 0 for legitimate
    
    customer = relationship("Customer", back_populates="transactions")
    investigation = relationship("InvestigationLog", back_populates="transaction", uselist=False)

class FraudCase(Base):
    __tablename__ = 'fraud_cases'

    case_id = Column(String, primary_key=True)
    transaction_id = Column(String, ForeignKey('transactions.transaction_id'))
    description = Column(Text)
    severity = Column(String) # LOW, MEDIUM, HIGH, CRITICAL
    resolution = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class InvestigationLog(Base):
    __tablename__ = 'investigation_logs'

    log_id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String, ForeignKey('transactions.transaction_id'))
    fraud_probability = Column(Float)
    anomaly_score = Column(Float)
    risk_level = Column(String)
    ml_explanation = Column(Text) # JSON string of SHAP values
    sql_evidence = Column(Text)
    rag_evidence = Column(Text)
    summary = Column(Text)
    recommended_action = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    transaction = relationship("Transaction", back_populates="investigation")
    decisions = relationship("Decision", back_populates="investigation")

class Decision(Base):
    __tablename__ = 'decisions'

    decision_id = Column(Integer, primary_key=True, autoincrement=True)
    log_id = Column(Integer, ForeignKey('investigation_logs.log_id'))
    human_decision = Column(String) # BLOCK, APPROVE, REQUEST VERIFICATION
    decided_by = Column(String)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    investigation = relationship("InvestigationLog", back_populates="decisions")
