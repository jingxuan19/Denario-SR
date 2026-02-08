import os
from denario import Denario, Journal

# llama 3.2 does not format returns properly
# llama 3.1:8b does not call functions properly
# deepseek-r1:8b can't do function calls
# qwen2.5:7b works, but slow
# Gemini uses a different Google API


den = Denario()

data_description = r"""
Write a short paper on harmonic oscillators. Generate several plots. Generate some data, which should not take more than 3 minutes to generate. 
"""

den.set_data_description(data_description = data_description)
den.show_data_description()

den.get_idea(llm="ollama/llama3.2")
den.show_idea()

den.get_method(llm="ollama/llama3.2")
den.show_method()

den.get_results(
    engineer_model="groq/llama-3.1-8b-instant",
    researcher_model="groq/llama-3.1-8b-instant",
    planner_model="groq/llama-3.1-8b-instant",
    plan_reviewer_model="groq/llama-3.1-8b-instant",
    orchestration_model="groq/llama-3.1-8b-instant",
    formatter_model="ollama/llama3.2",
    # restart_at_step=2
)

den.get_results(
    engineer_model="ollama/qwen3:8b",
    researcher_model="ollama/qwen3:8b",
    planner_model="ollama/qwen3:8b",
    plan_reviewer_model="ollama/qwen3:8b",
    orchestration_model="ollama/qwen3:8b",
    formatter_model="ollama/qwen3:8b",
    restart_at_step=4,
)

den.show_results()

den.get_sr(
    SR_module="pysr",
    data_dir="data",
    target_column="y",
    feature_columns=["time"]
)

den.get_paper(journal=Journal.APS,
              llm="ollama/qwen3:8b",
              add_citations=False
              )