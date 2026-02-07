#!/bin/bash
# Quick-start script for interactive testing with MCP clients

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                                                        ║${NC}"
echo -e "${BLUE}║        Agentique Interactive Testing Setup            ║${NC}"
echo -e "${BLUE}║                                                        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}✗ Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker is running${NC}"
echo ""

# Start the Docker Compose stack
echo -e "${YELLOW}Starting Docker Compose stack...${NC}"
docker compose -f docker-compose.interactive.yml up -d --wait

# Wait a bit for services to fully initialize
echo -e "${YELLOW}Waiting for services to be ready...${NC}"
sleep 3

# Check MCP server health
echo -e "${YELLOW}Checking MCP server health...${NC}"
if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ MCP server is healthy${NC}"
else
    echo -e "${RED}✗ MCP server is not responding${NC}"
    echo -e "${YELLOW}Checking logs:${NC}"
    docker logs agentique-interactive-mcp-server --tail 20
    exit 1
fi

# Check demo agent
echo -e "${YELLOW}Checking demo agent...${NC}"
if curl -sf http://localhost:9000/.well-known/agent-card.json > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Demo agent is healthy${NC}"
else
    echo -e "${RED}✗ Demo agent is not responding${NC}"
    echo -e "${YELLOW}Checking logs:${NC}"
    docker logs agentique-interactive-demo-agent --tail 20
    exit 1
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                                                        ║${NC}"
echo -e "${GREEN}║              Services are ready! 🎉                    ║${NC}"
echo -e "${GREEN}║                                                        ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Show connection information
echo -e "${BLUE}Connection Information:${NC}"
echo -e "  MCP Server:  http://localhost:8000"
echo -e "  Demo Agent:  http://localhost:9000"
echo ""

echo -e "${BLUE}Quick Tests:${NC}"
echo -e "  Health:      curl http://localhost:8000/health"
echo -e "  Agent card:  curl http://localhost:9000/.well-known/agent-card.json"
echo ""

# Show MCP client configuration instructions
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}MCP Client Configuration${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${BLUE}Claude CLI:${NC}"
echo -e "  1. Edit: ~/.claude/mcp_settings.json"
echo -e "  2. Add config from: examples/mcp-clients/claude-cli-config.json"
echo -e "  3. Test: ${GREEN}claude \"List available agents\"${NC}"
echo ""
echo -e "${BLUE}Claude Desktop:${NC}"
echo -e "  macOS:   ~/Library/Application Support/Claude/claude_desktop_config.json"
echo -e "  Windows: %APPDATA%\\Claude\\claude_desktop_config.json"
echo -e "  Linux:   ~/.config/Claude/claude_desktop_config.json"
echo -e "  Config:  See examples/mcp-clients/claude-desktop-config.json"
echo ""
echo -e "${BLUE}VS Code (Cline/Continue):${NC}"
echo -e "  1. Open workspace settings (.vscode/settings.json)"
echo -e "  2. Add config from: examples/mcp-clients/vscode-settings.json"
echo -e "  3. Restart VS Code"
echo ""

# Show demo scenarios menu
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Demo Scenarios (try these in your MCP client)${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${GREEN}1. Get Help${NC}"
echo -e "   Send to demo agent: ${YELLOW}/help${NC}"
echo ""
echo -e "${GREEN}2. Echo Test${NC}"
echo -e "   Send to demo agent: ${YELLOW}/echo Hello, world!${NC}"
echo ""
echo -e "${GREEN}3. Streaming Response${NC}"
echo -e "   Send to demo agent: ${YELLOW}/stream Tell me a story${NC}"
echo ""
echo -e "${GREEN}4. Calculator${NC}"
echo -e "   Send to demo agent: ${YELLOW}/calc 15 * 7${NC}"
echo ""
echo -e "${GREEN}5. Multi-turn Conversation${NC}"
echo -e "   Send: ${YELLOW}Remember my name is Alice${NC}"
echo -e "   Then: ${YELLOW}What did I just tell you?${NC}"
echo ""
echo -e "${GREEN}6. Error Handling${NC}"
echo -e "   Send to demo agent: ${YELLOW}/error${NC}"
echo ""
echo -e "${GREEN}7. Background Task${NC}"
echo -e "   Send to demo agent: ${YELLOW}/background process data${NC}"
echo ""
echo -e "${GREEN}8. View Memory${NC}"
echo -e "   Send to demo agent: ${YELLOW}/memory${NC}"
echo ""

echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Helpful Commands${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  View logs:        ${GREEN}docker compose -f docker-compose.interactive.yml logs -f${NC}"
echo -e "  Stop services:    ${GREEN}docker compose -f docker-compose.interactive.yml down${NC}"
echo -e "  Restart:          ${GREEN}docker compose -f docker-compose.interactive.yml restart${NC}"
echo -e "  Show status:      ${GREEN}docker compose -f docker-compose.interactive.yml ps${NC}"
echo ""
echo -e "${BLUE}Documentation:${NC}"
echo -e "  Full guide:       ${GREEN}cat INTERACTIVE_TESTING.md${NC}"
echo -e "  MCP configs:      ${GREEN}ls examples/mcp-clients/${NC}"
echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${GREEN}Ready to test! Configure your MCP client and start chatting.${NC}"
echo -e "${BLUE}See INTERACTIVE_TESTING.md for detailed test scenarios.${NC}"
echo ""
