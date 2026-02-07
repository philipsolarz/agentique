# Dockerfile for the demo A2A agent
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    a2a-sdk>=0.3.22 \
    fastapi \
    uvicorn \
    pydantic

# Copy agent code
COPY tests/agents/demo_agent.py /app/demo_agent.py
COPY tests/agents/helpers.py /app/helpers.py

# Create server script
RUN echo 'from demo_agent import DemoAgent\n\
from helpers import create_mock_a2a_app\n\
import uvicorn\n\
import os\n\
\n\
if __name__ == "__main__":\n\
    agent = DemoAgent()\n\
    app = create_mock_a2a_app(\n\
        agent,\n\
        name="demo-agent",\n\
        streaming=True,\n\
        push_notifications=False\n\
    )\n\
    \n\
    host = os.getenv("A2A_HOST", "0.0.0.0")\n\
    port = int(os.getenv("A2A_PORT", "9000"))\n\
    \n\
    print(f"Starting demo A2A agent on {host}:{port}")\n\
    uvicorn.run(app, host=host, port=port)\n\
' > /app/server.py

EXPOSE 9000

CMD ["python", "server.py"]
