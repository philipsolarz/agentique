FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md /app/
COPY src /app/src

RUN uv venv --python /usr/local/bin/python3.12 && uv sync --extra ui

ENV AGENTIQUE_TRANSPORT=http
ENV AGENTIQUE_HOST=0.0.0.0
ENV AGENTIQUE_PORT=8000

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "python", "-m", "agentique"]
