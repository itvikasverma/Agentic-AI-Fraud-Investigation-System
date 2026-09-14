from langchain_core.tools import tool
from database.connection import SessionLocal
from database.models import Customer, Transaction, FraudCase
from sqlalchemy import text

def is_safe_query(query: str) -> bool:
    """Basic SQL injection/mutation protection."""
    dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'GRANT', 'REVOKE']
    query_upper = query.upper()
    for kw in dangerous_keywords:
        if kw in query_upper:
            return False
    return True

@tool
def execute_read_only_sql(query: str) -> str:
    """
    Executes a READ-ONLY SQL query on the fraud database.
    Useful for checking customer history, calculating transaction frequencies, 
    and finding averages. 
    Tables available:
    - customers (customer_id, name, account_age_days, average_transaction, transaction_frequency_per_month, usual_locations, usual_devices, previous_fraud_count)
    - transactions (transaction_id, customer_id, amount, timestamp, location, device_id, merchant_category, transaction_hour, weekend, location_change, device_change, international_transaction, failed_attempts, distance_from_previous_transaction, transaction_velocity, amount_deviation, status, fraud_label)
    - fraud_cases (case_id, transaction_id, description, severity, resolution, created_at)
    """
    if not is_safe_query(query):
        return "ERROR: Unsafe query detected. Only SELECT statements are allowed."
    
    db = SessionLocal()
    try:
        # Wrap the query in text()
        result = db.execute(text(query)).fetchall()
        if not result:
            return "No results found."
        
        # Convert to list of dicts for string representation
        # Getting column names from the result
        rows = [dict(row._mapping) for row in result]
        return str(rows[:10]) # Return max 10 rows to prevent context overflow
    except Exception as e:
        return f"SQL Error: {str(e)}"
    finally:
        db.close()
