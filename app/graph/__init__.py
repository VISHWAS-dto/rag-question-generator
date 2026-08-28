"""LangGraph graphs that wrap the project's LLM calls.

Each engine (question generation, follow-up decision, report analysis) still
exposes the same public function and return type it always has; internally it
now builds and invokes a small StateGraph from `repair_graph.py` that adds an
automatic parse/repair loop around the hosted model's raw-JSON response.
"""
