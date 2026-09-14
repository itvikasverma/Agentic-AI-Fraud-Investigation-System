# Device Security Policy

## Overview
Guidelines for device and location-based fraud detection.

## Rule 1: New Device Detection
Large transactions originating from a new device combined with unusual transaction timing should undergo additional verification.

## Rule 2: Impossible Travel
If the distance from the previous transaction implies a travel speed greater than 800 km/h, the transaction is likely fraudulent (Impossible Travel).

## Rule 3: International Transactions
If the customer has no history of international transactions, and a new international transaction occurs, it should be marked as MEDIUM risk and require user confirmation if it exceeds $100.
