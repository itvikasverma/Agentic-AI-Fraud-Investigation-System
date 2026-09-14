import os
import random
import uuid
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

from faker import Faker
from .connection import init_db, SessionLocal
from .models import Customer, Transaction, FraudCase

fake = Faker()

NUM_CUSTOMERS = 1000
NUM_TRANSACTIONS = 50000
FRAUD_RATE = 0.05

def generate_customers():
    customers = []
    for _ in range(NUM_CUSTOMERS):
        customers.append({
            "customer_id": str(uuid.uuid4()),
            "name": fake.name(),
            "account_age_days": random.randint(1, 3650),
            "average_transaction": round(random.uniform(10, 500), 2),
            "transaction_frequency_per_month": random.randint(1, 50),
            "usual_locations": json.dumps([fake.city() for _ in range(random.randint(1, 3))]),
            "usual_devices": json.dumps([str(uuid.uuid4())[:8] for _ in range(random.randint(1, 3))]),
            "previous_fraud_count": random.choices([0, 1, 2], weights=[0.95, 0.04, 0.01])[0]
        })
    return pd.DataFrame(customers)

def generate_transactions(customers_df):
    transactions = []
    merchants = ['RETAIL', 'GROCERY', 'TRAVEL', 'ELECTRONICS', 'ONLINE_SERVICE', 'RESTAURANT', 'UTILITY']
    
    start_date = datetime.now() - timedelta(days=90)
    
    for i in range(NUM_TRANSACTIONS):
        customer = customers_df.sample().iloc[0]
        is_fraud = random.random() < FRAUD_RATE
        
        # Base values
        amount = random.gauss(customer['average_transaction'], customer['average_transaction'] * 0.2)
        transaction_hour = random.randint(0, 23)
        location_change = random.random() < 0.1
        device_change = random.random() < 0.1
        international_transaction = random.random() < 0.05
        failed_attempts = random.choices([0, 1, 2, 3], weights=[0.9, 0.05, 0.03, 0.02])[0]
        distance_from_previous_transaction = random.expovariate(1/10) # avg 10 km
        transaction_velocity = random.randint(1, 3)
        
        # Correlate features with fraud
        if is_fraud:
            amount *= random.uniform(3, 10) # Much higher amount
            transaction_hour = random.choices([0, 1, 2, 3, 4, 22, 23], k=1)[0]
            location_change = random.random() < 0.8
            device_change = random.random() < 0.7
            international_transaction = random.random() < 0.4
            failed_attempts = random.randint(2, 5)
            distance_from_previous_transaction = random.uniform(500, 5000)
            transaction_velocity = random.randint(5, 15)
        
        amount = max(1.0, round(amount, 2))
        amount_deviation = round((amount - customer['average_transaction']) / max(1, customer['average_transaction']), 4)
        
        timestamp = start_date + timedelta(
            days=random.randint(0, 90),
            hours=transaction_hour,
            minutes=random.randint(0, 59)
        )
        weekend = timestamp.weekday() >= 5
        
        transactions.append({
            "transaction_id": str(uuid.uuid4()),
            "customer_id": customer['customer_id'],
            "amount": amount,
            "timestamp": timestamp,
            "location": fake.city() if location_change else eval(customer['usual_locations'])[0],
            "device_id": str(uuid.uuid4())[:8] if device_change else eval(customer['usual_devices'])[0],
            "merchant_category": random.choice(merchants),
            "transaction_hour": transaction_hour,
            "weekend": weekend,
            "location_change": location_change,
            "device_change": device_change,
            "international_transaction": international_transaction,
            "failed_attempts": failed_attempts,
            "distance_from_previous_transaction": round(distance_from_previous_transaction, 2),
            "transaction_velocity": transaction_velocity,
            "amount_deviation": amount_deviation,
            "status": "APPROVED" if not is_fraud else "REVIEW",
            "fraud_label": is_fraud
        })
        
        # Print progress
        if (i+1) % 10000 == 0:
            print(f"Generated {i+1} transactions...")
            
    return pd.DataFrame(transactions)

def seed_db():
    print("Initializing Database...")
    init_db()
    db = SessionLocal()
    
    # Check if data already exists
    if db.query(Customer).count() > 0:
        print("Database already seeded. Skipping.")
        db.close()
        return

    import json
    
    print("Generating Customers...")
    customers_df = generate_customers()
    
    print("Generating Transactions...")
    transactions_df = generate_transactions(customers_df)
    
    # Save to CSV for ML training
    os.makedirs('data', exist_ok=True)
    transactions_df.to_csv('data/transactions.csv', index=False)
    customers_df.to_csv('data/customers.csv', index=False)
    print("Saved datasets to data/")
    
    print("Seeding database...")
    
    # Batch insert customers
    customer_objects = []
    for _, row in customers_df.iterrows():
        c = Customer(
            customer_id=row['customer_id'],
            name=row['name'],
            account_age_days=row['account_age_days'],
            average_transaction=row['average_transaction'],
            transaction_frequency_per_month=row['transaction_frequency_per_month'],
            usual_locations=row['usual_locations'],
            usual_devices=row['usual_devices'],
            previous_fraud_count=row['previous_fraud_count']
        )
        customer_objects.append(c)
    
    db.bulk_save_objects(customer_objects)
    db.commit()
    print("Customers seeded.")
    
    # Batch insert transactions
    transaction_objects = []
    for _, row in transactions_df.iterrows():
        t = Transaction(
            transaction_id=row['transaction_id'],
            customer_id=row['customer_id'],
            amount=row['amount'],
            timestamp=row['timestamp'],
            location=row['location'],
            device_id=row['device_id'],
            merchant_category=row['merchant_category'],
            transaction_hour=row['transaction_hour'],
            weekend=row['weekend'],
            location_change=row['location_change'],
            device_change=row['device_change'],
            international_transaction=row['international_transaction'],
            failed_attempts=row['failed_attempts'],
            distance_from_previous_transaction=row['distance_from_previous_transaction'],
            transaction_velocity=row['transaction_velocity'],
            amount_deviation=row['amount_deviation'],
            status=row['status'],
            fraud_label=row['fraud_label']
        )
        transaction_objects.append(t)
    
    # Split into chunks of 10000 for insertion
    chunk_size = 10000
    for i in range(0, len(transaction_objects), chunk_size):
        db.bulk_save_objects(transaction_objects[i:i+chunk_size])
        db.commit()
        print(f"Seeded {min(i+chunk_size, len(transaction_objects))} transactions.")
        
    print("Generating Historical Fraud Cases...")
    # Add a few fraud cases
    fraud_txs = db.query(Transaction).filter(Transaction.fraud_label == True).limit(50).all()
    case_objects = []
    for i, tx in enumerate(fraud_txs):
        case = FraudCase(
            case_id=f"CASE-{i+1:03d}",
            transaction_id=tx.transaction_id,
            description=f"Detected unusual transaction amount from a new device in {tx.location}.",
            severity=random.choice(["HIGH", "CRITICAL"]),
            resolution="Confirmed fraud with customer. Card blocked."
        )
        case_objects.append(case)
    db.bulk_save_objects(case_objects)
    db.commit()
        
    db.close()
    print("Database seeding complete!")

if __name__ == "__main__":
    seed_db()
