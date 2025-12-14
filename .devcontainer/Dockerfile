FROM python:3.12-slim

# Install Node.js, npm, curl, and git
RUN apt-get update && apt-get install -y \
    curl \
    nodejs \
    npm \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Claude Code
RUN npm install -g @anthropic-ai/claude-code

# Install GitHub CLI
RUN apt-get update && apt-get install -y gh && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace