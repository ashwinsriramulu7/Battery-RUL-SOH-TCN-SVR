# models/__init__.py
"""
Model training and evaluation package for NASA battery RUL/SOH.

Contains:
- TCN for sequence → SOH trajectory modeling
- Piecewise SVR for RUL trajectories, trained on TCN representations
"""
