#!/usr/bin/env bash
set -e
echo "🔧 Setting up SQLMind..."
python -m venv .venv
source .venv/bin/activate
pip install -e ".[db,graph,dev]"
cp .env.example .env
echo "✅ Edit .env with your API keys, then:"
echo "   pytest tests/ -v"
echo "   python sqlmind_graph.py inspect examples/schema.sqlmind.yaml"
echo "   python integrations/sqlmind_openai_agents.py --demo"
echo "   python integrations/sqlmind_langgraph.py --demo"
