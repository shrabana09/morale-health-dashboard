"""
config.py
---------
Loads all configuration from the .env file. This is the ONLY place
environment variables are read from — every other module imports
from here instead of calling os.getenv() directly, so secrets never
get hardcoded or scattered across the codebase.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads the .env file in the project root

# --- Groq / AI agent settings -------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.3"))
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "1024"))

# --- Database -------------------------------------------------------------
DB_PATH = os.getenv("DB_PATH", "morale.db")

# --- Sampling ---------------------------------------------------------------
# How many example comments get sent to the AI agent alongside the
# aggregated stats. Kept small on purpose — see agent.py.
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "15"))

if not GROQ_API_KEY:
    # Don't crash on import (e.g. during local dev before .env is filled in),
    # but make it obvious in logs that the agent call will fail later.
    print("[config] WARNING: GROQ_API_KEY is not set. Add it to your .env file.")
