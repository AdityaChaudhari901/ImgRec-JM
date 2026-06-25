"""Accuracy evaluation harness for the grocery dispute endpoint.

Two modes share one set of metrics:
- engine mode: feed labelled observations + shipment through the deterministic
  classifier + decision engine (offline, no API cost) — proves business-logic
  accuracy and acts as a golden regression set.
- e2e mode: run real images through the Gemini observation call, then the same
  engine — drop in real labelled JioMart photos to get the model-level number.
"""
