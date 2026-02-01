from pydantic import BaseModel
from typing import Dict

class LLM(BaseModel):
    """LLM base model"""
    name: str
    """Name/identifier of the model."""
    max_output_tokens: int
    """Maximum output tokens allowed."""
    temperature: float | None
    """Temperature of the model."""

gemini20flash = LLM(name="gemini-2.0-flash",
                    max_output_tokens=8192,
                    temperature=0.7)
"""`gemini-2.0-flash` model."""

gemini25flash = LLM(name="gemini-2.5-flash",
                    max_output_tokens=65536,
                    temperature=0.7)
"""`gemini-2.5-flash` model."""

gemini25pro = LLM(name="gemini-2.5-pro",
                  max_output_tokens=65536,
                  temperature=0.7)
"""`gemini-2.5-pro` model."""

o3mini = LLM(name="o3-mini-2025-01-31",
             max_output_tokens=100000,
             temperature=None)
"""`o3-mini` model."""

gpt4o = LLM(name="gpt-4o-2024-11-20",
            max_output_tokens=16384,
            temperature=0.5)
"""`gpt-4o` model."""

gpt41 = LLM(name="gpt-4.1-2025-04-14",
            max_output_tokens=16384,
            temperature=0.5)
"""`gpt-4.1` model."""

gpt41mini = LLM(name="gpt-4.1-mini",
                max_output_tokens=16384,
                temperature=0.5)
"""`gpt-4.1-mini` model."""

gpt4omini = LLM(name="gpt-4o-mini-2024-07-18",
                max_output_tokens=16384,
                temperature=0.5)
"""`gpt-4o-mini` model."""

gpt45 = LLM(name="gpt-4.5-preview-2025-02-27",
            max_output_tokens=16384,
            temperature=0.5)
"""`gpt-4.5-preview` model."""

gpt5 = LLM(name="gpt-5",
           max_output_tokens=128000,
           temperature=None)
"""`gpt-5` model """

gpt5mini = LLM(name="gpt-5-mini",
               max_output_tokens=128000,
               temperature=None)
"""`gpt-5-mini` model."""

claude37sonnet = LLM(name="claude-3-7-sonnet-20250219",
                     max_output_tokens=64000,
                     temperature=0)
"""`claude-3-7-sonnet` model."""

claude4opus = LLM(name="claude-opus-4-20250514",
                   max_output_tokens=32000,
                   temperature=0)
"""`claude-4-Opus` model."""

claude41opus = LLM(name="claude-opus-4-1-20250805",
                   max_output_tokens=32000,
                   temperature=0)
"""`claude-4.1-Opus` model."""

# =============================================================================
# Local Ollama Models
# =============================================================================

ollama_llama32 = LLM(name="ollama/llama3.2",
                     max_output_tokens=8192,
                     temperature=0.7)
"""Llama 3.2 via Ollama (local)."""

ollama_llama31_8b = LLM(name="ollama/llama3.1:8b",
                        max_output_tokens=8192,
                        temperature=0.7)
"""Llama 3.1 8B via Ollama (local)."""

ollama_llama33_70b = LLM(name="ollama/llama3.3:70b",
                         max_output_tokens=8192,
                         temperature=0.7)
"""Llama 3.3 70B via Ollama (local)."""

ollama_qwen25_7b = LLM(name="ollama/qwen2.5:7b",
                       max_output_tokens=8192,
                       temperature=0.7)
"""Qwen 2.5 7B via Ollama (local)."""

ollama_qwen25_14b = LLM(name="ollama/qwen2.5:14b",
                        max_output_tokens=8192,
                        temperature=0.7)
"""Qwen 2.5 14B via Ollama (local)."""

ollama_qwen25_coder = LLM(name="ollama/qwen2.5-coder:7b",
                          max_output_tokens=8192,
                          temperature=0.7)
"""Qwen 2.5 Coder 7B via Ollama (local)."""

ollama_deepseek_r1_8b = LLM(name="ollama/deepseek-r1:8b",
                            max_output_tokens=16384,
                            temperature=0.7)
"""DeepSeek R1 8B via Ollama (local)."""

ollama_deepseek_r1_14b = LLM(name="ollama/deepseek-r1:14b",
                             max_output_tokens=16384,
                             temperature=0.7)
"""DeepSeek R1 14B via Ollama (local)."""

ollama_mistral = LLM(name="ollama/mistral",
                     max_output_tokens=8192,
                     temperature=0.7)
"""Mistral via Ollama (local)."""

ollama_phi4 = LLM(name="ollama/phi4",
                  max_output_tokens=16384,
                  temperature=0.7)
"""Phi-4 via Ollama (local)."""

ollama_gemma2_9b = LLM(name="ollama/gemma2:9b",
                       max_output_tokens=8192,
                       temperature=0.7)
"""Gemma 2 9B via Ollama (local)."""

# =============================================================================
# Groq Models (Free tier with excellent function calling)
# =============================================================================

groq_llama33_70b = LLM(name="groq/llama-3.3-70b-versatile",
                       max_output_tokens=32768,
                       temperature=0.7)
"""Llama 3.3 70B via Groq (free tier)."""

groq_llama31_70b = LLM(name="groq/llama-3.1-70b-versatile",
                       max_output_tokens=32768,
                       temperature=0.7)
"""Llama 3.1 70B via Groq (free tier)."""

groq_llama31_8b = LLM(name="groq/llama-3.1-8b-instant",
                      max_output_tokens=8192,
                      temperature=0.7)
"""Llama 3.1 8B via Groq (free tier, fastest)."""

groq_mixtral = LLM(name="groq/mixtral-8x7b-32768",
                   max_output_tokens=32768,
                   temperature=0.7)
"""Mixtral 8x7B via Groq (free tier)."""

models : Dict[str, LLM] = {
                            "gemini-2.0-flash" : gemini20flash,
                            "gemini-2.5-flash" : gemini25flash,
                            "gemini-2.5-pro" : gemini25pro,
                            "o3-mini" : o3mini,
                            "gpt-4o" : gpt4o,
                            "gpt-4.1" : gpt41,
                            "gpt-4.1-mini" : gpt41mini,
                            "gpt-4o-mini" : gpt4omini,
                            "gpt-4.5" : gpt45,
                            "gpt-5" : gpt5,
                            "gpt-5-mini" : gpt5mini,
                            "claude-3.7-sonnet" : claude37sonnet,
                            "claude-4-opus" : claude4opus,
                            "claude-4.1-opus" : claude41opus,
                            # Ollama
                            "ollama/llama3.2" : ollama_llama32,
                            "ollama/llama3.1:8b" : ollama_llama31_8b,
                            "ollama/llama3.3:70b" : ollama_llama33_70b,
                            "ollama/qwen2.5:7b" : ollama_qwen25_7b,
                            "ollama/qwen2.5:14b" : ollama_qwen25_14b,
                            "ollama/qwen2.5-coder:7b" : ollama_qwen25_coder,
                            "ollama/deepseek-r1:8b" : ollama_deepseek_r1_8b,
                            "ollama/deepseek-r1:14b" : ollama_deepseek_r1_14b,
                            "ollama/mistral" : ollama_mistral,
                            "ollama/phi4" : ollama_phi4,
                            "ollama/gemma2:9b" : ollama_gemma2_9b,
                            # Groq (free tier)
                            "groq/llama-3.3-70b-versatile" : groq_llama33_70b,
                            "groq/llama-3.1-70b-versatile" : groq_llama31_70b,
                            "groq/llama-3.1-8b-instant" : groq_llama31_8b,
                            "groq/mixtral-8x7b-32768" : groq_mixtral,
                           }
"""Dictionary with the available models."""
