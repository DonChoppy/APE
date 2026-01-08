FROM python:3.11-slim

# Install system dependencies for OS-level operations (curl, git, etc.)
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install MCP SDK and transport dependencies
# Note: 'mcp' is the official package name as confirmed.
RUN pip install --no-cache-dir \
    mcp \
    starlette \
    uvicorn \
    httpx

# Create a restricted working directory for file operations
RUN mkdir -p /app/workdir
ENV ALLOWED_ROOT=/app/workdir

# Copy the server implementation
COPY server.py .

# Expose the port
EXPOSE 8080

# Run the server
CMD ["python", "server.py"]
