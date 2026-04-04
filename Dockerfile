FROM ghcr.io/meta-pytorch/openenv-base:latest AS builder

WORKDIR /app/env

# Install git for git+ dependencies
RUN apt-get update && apt-get install -y --no-install-recommends git gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency spec first for layer caching
COPY pyproject.toml ./
COPY requirements.txt ./

# Install all dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy full source
COPY . .

# Validate environment loads cleanly at build time
RUN python -c "from env.environment import IndiaITREnvironment; e = IndiaITREnvironment('task1_parse'); e.reset(); print('ENV OK')"

FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app/env

# Copy installed packages and source from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /app/env /app/env

ENV PYTHONPATH="/app/env:$PYTHONPATH"
ENV ENABLE_WEB_INTERFACE=true

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

EXPOSE 7860

CMD ["python", "app.py"]
