# Fraud Policy

## Overview
This document outlines the core fraud policies for transaction monitoring.

## Rule 1: High Amount Deviation
Transactions that exceed 300% of the customer's average transaction amount must be flagged for review. If the transaction exceeds 500%, it must be blocked immediately pending verification.

## Rule 2: Multiple Failures
More than 3 failed attempts in the last 24 hours indicates potential account takeover or card testing. Subsequent successful transactions must be reviewed.

## Rule 3: Velocity
More than 5 transactions in a single hour from a single customer is considered high velocity and suspicious.
