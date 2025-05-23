#!/bin/bash

# Enable verbose mode to see model and request info without all the noise
export LOGLEVEL=INFO
#export LOGLEVEL=DEBUG
#export SHOW_MODEL_DETAILS=true

# Run server with preferred provider and models

# Using o3 with custom system prompt experiment

# Change Preferred Provider to together
PREFERRED_PROVIDER=together uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload

# Change Preferred Provider to openai
# PREFERRED_PROVIDER=openai BIG_MODEL=o3 SMALL_MODEL=o4-mini uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload


