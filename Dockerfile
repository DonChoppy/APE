# Final, simplified Dockerfile for APE
FROM python:3.11-slim-bookworm

WORKDIR /app

# Install system dependencies including gosu for privilege-dropping
RUN apt-get update && apt-get install -y \
    curl \
    gdb \
    strace \
    procps \
    util-linux \
    gosu \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
# Create dummy package to satisfy setuptools during dependency installation
RUN mkdir ape && touch ape/__init__.py
# Since faiss-cpu is a pure-python package, no special build steps are needed.
RUN pip install --no-cache-dir .'[llm,cli]'

# Copy the rest of the code
COPY . .

# Add and configure the entrypoint script
COPY scripts/docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Create non-root user and its home directory
RUN groupadd -r apeuser && useradd -r -g apeuser -d /home/apeuser -m apeuser

# Set the entrypoint, which will run as root
ENTRYPOINT ["docker-entrypoint.sh"]

# Default command (will be passed to the entrypoint)
CMD ["python", "mcp_server.py"]