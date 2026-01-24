FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md /app/
COPY src /app/src

RUN uv sync

ENV AGENTIQUE_TRANSPORT=http
ENV AGENTIQUE_HOST=0.0.0.0
ENV AGENTIQUE_PORT=8000

EXPOSE 8000

CMD ["uv", "run", "python", "-m", "agentique"]
