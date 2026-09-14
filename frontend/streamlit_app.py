import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

API_URL = f"http://localhost:{os.getenv('API_PORT', 8000)}"

st.set_page_config(page_title="Fraud Investigation Center", page_icon="🏦", layout="wide")

st.title("🏦 Fraud Investigation Center")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Investigate Transaction", "Investigation History", "Fraud Analytics", "System/Agent Trace"])

if page == "Investigate Transaction":
    st.header("🔍 Investigate Transaction")
    
    col1, col2 = st.columns(2)
    with col1:
        tx_id = st.text_input("Transaction ID", value="TX-999")
        cust_id = st.text_input("Customer ID", value="CUST-123")
        amount = st.number_input("Amount (₹)", value=85000.0)
        location = st.text_input("Location", value="Delhi")
        device_id = st.text_input("Device ID", value="DEV-NEW-88")
    
    with col2:
        tx_hour = st.number_input("Transaction Hour", value=3, min_value=0, max_value=23)
        location_change = st.checkbox("Location Change", value=True)
        device_change = st.checkbox("Device Change", value=True)
        international = st.checkbox("International Transaction", value=False)
        failed_attempts = st.number_input("Failed Attempts", value=0)
        vel = st.slider("Transaction Velocity (per hour)", min_value=1.0, max_value=20.0, value=8.0)
        
    # Hidden defaults for demo
    weekend = False
    dist = 2500.0
    amt_dev = 28.3 # (85000 - 3000)/3000
    
    if st.button("🔍 Investigate Transaction", type="primary"):
        with st.spinner("Agents are investigating..."):
            payload = {
                "transaction_id": tx_id,
                "customer_id": cust_id,
                "amount": amount,
                "location": location,
                "device_id": device_id,
                "merchant_category": "ELECTRONICS",
                "transaction_hour": tx_hour,
                "weekend": weekend,
                "location_change": location_change,
                "device_change": device_change,
                "international_transaction": international,
                "failed_attempts": failed_attempts,
                "distance_from_previous_transaction": dist,
                "transaction_velocity": vel,
                "amount_deviation": amt_dev
            }
            
            try:
                response = requests.post(f"{API_URL}/api/investigate", json=payload)
                response.raise_for_status()
                data = response.json()
                
                # Store in session state to persist
                st.session_state['investigation_data'] = data
                st.session_state['payload'] = payload
                
            except requests.exceptions.RequestException as e:
                st.error(f"API Error: {e}")

    # Display Results if available
    if 'investigation_data' in st.session_state:
        data = st.session_state['investigation_data']
        payload = st.session_state['payload']
        
        st.markdown("---")
        st.header("Risk Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        prob = data['fraud_probability'] * 100
        prob_color = "red" if prob > 70 else "orange" if prob > 30 else "green"
        col1.metric("Fraud Probability", f"{prob:.1f}%")
        
        col2.metric("Anomaly Score", f"{data['anomaly_score']:.2f}")
        
        risk = data['risk_level']
        col3.metric("Risk Level", risk)
        
        col4.metric("Recommended Action", data['recommended_action'])
        
        st.markdown("---")
        
        # Tabs for detailed information
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["ML Explanation", "Customer Behavior", "RAG Evidence", "Investigation Report", "Agent Trace"])
        
        with tab1:
            st.subheader("SHAP Feature Contributions")
            if data['ml_explanation']:
                df_shap = pd.DataFrame(data['ml_explanation'])
                fig = px.bar(df_shap, x='contribution', y='feature', orientation='h', 
                             color='contribution', color_continuous_scale=px.colors.diverging.RdBu)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No ML explanation available.")
                
        with tab2:
            st.subheader("Transaction Details")
            st.json(payload)
            
        with tab3:
            st.subheader("Retrieved Knowledge Base Evidence")
            # The RAG evidence is in the state but the API currently doesn't return the raw rag_evidence strings.
            # We will show the investigation summary which synthesized it.
            st.info("Agent retrieved policies and historical cases to synthesize the report.")
            
        with tab4:
            st.subheader("Investigation Report")
            if data.get('investigation_summary'):
                st.markdown(data['investigation_summary'])
            else:
                st.info("Report not generated for low risk transactions.")
                
        with tab5:
            st.subheader("Agent Execution Trace")
            for step in data['agent_trace']:
                with st.expander(f"{step['timestamp']} - {step['agent_name']}: {step['action']}"):
                    st.write(f"**Status:** {step['status']}")
                    st.write(f"**Result:** {step['short_result']}")
        
        # Human in the loop
        if data['requires_human_review']:
            st.markdown("---")
            st.error("⚠️ HUMAN REVIEW REQUIRED")
            
            h_col1, h_col2, h_col3 = st.columns(3)
            with h_col1:
                if st.button("BLOCK", use_container_width=True):
                    requests.post(f"{API_URL}/api/decision", json={"transaction_id": data['transaction_id'], "decision": "BLOCK"})
                    st.success("Decision Recorded: BLOCK")
            with h_col2:
                if st.button("APPROVE", use_container_width=True):
                    requests.post(f"{API_URL}/api/decision", json={"transaction_id": data['transaction_id'], "decision": "APPROVE"})
                    st.success("Decision Recorded: APPROVE")
            with h_col3:
                if st.button("REQUEST VERIFICATION", use_container_width=True):
                    requests.post(f"{API_URL}/api/decision", json={"transaction_id": data['transaction_id'], "decision": "REQUEST VERIFICATION"})
                    st.success("Decision Recorded: REQUEST VERIFICATION")

elif page == "Dashboard":
    st.header("System Overview")
    st.info("Use the sidebar to navigate to the Investigation tool.")

elif page == "System/Agent Trace":
    st.header("System Logs & Traces")
    if 'investigation_data' in st.session_state:
        st.json(st.session_state['investigation_data']['agent_trace'])
    else:
        st.info("Run an investigation first to see traces.")
        
elif page == "Investigation History":
    st.header("📋 Investigation History & Decisions")
    st.write("This page shows the latest human decisions recorded in the database.")
    
    if st.button("🔄 Refresh History"):
        st.rerun()
        
    try:
        response = requests.get(f"{API_URL}/api/history")
        if response.status_code == 200:
            history_data = response.json()
            if history_data:
                df = pd.DataFrame(history_data)
                # Format columns
                df['fraud_probability'] = df['fraud_probability'].apply(lambda x: f"{x*100:.1f}%")
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Display table
                st.dataframe(
                    df[['transaction_id', 'timestamp', 'fraud_probability', 'risk_level', 'ai_recommendation', 'human_decision', 'decided_by']],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No decisions recorded yet.")
        else:
            st.error("Failed to load history from API.")
    except Exception as e:
        st.error(f"Could not connect to API: {e}")
        
else:
    st.info("Page under construction for demo purposes.")
