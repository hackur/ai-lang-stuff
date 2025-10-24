#!/bin/bash
# Test script to verify setup is working correctly
set -e

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

passed=0
failed=0

print_pass() {
    echo -e "${GREEN}✓ PASS${NC} $1"
    ((passed++))
}

print_fail() {
    echo -e "${RED}✗ FAIL${NC} $1"
    ((failed++))
}

print_test() {
    echo -e "${BOLD}Testing:${NC} $1"
}

echo -e "${BOLD}Running Setup Tests...${NC}\n"

# Test 1: Check if uv is installed
print_test "uv installation"
if command -v uv >/dev/null 2>&1; then
    print_pass "uv is installed ($(uv --version))"
else
    print_fail "uv not found"
fi

# Test 2: Check if node is installed
print_test "Node.js installation"
if command -v node >/dev/null 2>&1; then
    print_pass "Node.js is installed ($(node --version))"
else
    print_fail "Node.js not found"
fi

# Test 3: Check if Ollama is installed
print_test "Ollama installation"
if command -v ollama >/dev/null 2>&1; then
    print_pass "Ollama is installed"
else
    print_fail "Ollama not found"
fi

# Test 4: Check if Ollama server is running
print_test "Ollama server status"
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    print_pass "Ollama server is responding"
else
    print_fail "Ollama server not responding (is 'ollama serve' running?)"
fi

# Test 5: Check if models are available
print_test "Ollama models"
if ollama list | grep -q "qwen3"; then
    print_pass "Qwen3 models available"
else
    print_fail "No Qwen3 models found (run: ollama pull qwen3:8b)"
fi

# Test 6: Check Python environment
print_test "Python environment"
if uv run python -c "import sys; print(sys.version)" >/dev/null 2>&1; then
    version=$(uv run python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_pass "Python environment working (Python $version)"
else
    print_fail "Python environment not working"
fi

# Test 7: Check LangChain imports
print_test "LangChain installation"
if uv run python -c "from langchain_ollama import ChatOllama" 2>/dev/null; then
    print_pass "LangChain imports successful"
else
    print_fail "LangChain imports failed (run: uv sync)"
fi

# Test 8: Check directory structure
print_test "Directory structure"
if [ -d "examples" ] && [ -d "config" ] && [ -d "plans" ]; then
    print_pass "Required directories exist"
else
    print_fail "Missing required directories"
fi

# Test 9: Check .env file
print_test ".env file"
if [ -f ".env" ]; then
    print_pass ".env file exists"
else
    print_fail ".env file missing (run: cp config/.env.example .env)"
fi

# Test 10: Test actual model inference
print_test "Model inference"
test_result=$(uv run python -c "
from langchain_ollama import ChatOllama
try:
    llm = ChatOllama(model='qwen3:8b', base_url='http://localhost:11434')
    response = llm.invoke('Say OK')
    print('OK' if response.content else 'FAIL')
except Exception as e:
    print('FAIL')
" 2>/dev/null)

if [ "$test_result" = "OK" ]; then
    print_pass "Model inference working"
else
    print_fail "Model inference failed"
fi

# Summary
echo ""
echo -e "${BOLD}Test Summary:${NC}"
echo -e "  ${GREEN}Passed:${NC} $passed"
echo -e "  ${RED}Failed:${NC} $failed"
echo ""

if [ $failed -eq 0 ]; then
    echo -e "${BOLD}${GREEN}All tests passed!${NC} Setup is complete and working."
    echo ""
    echo "You can now:"
    echo "  • Run examples: uv run python examples/01-foundation/simple_chat.py"
    echo "  • Launch LangGraph Studio: npx langgraph@latest dev"
    echo "  • Read documentation: cat README.md"
    exit 0
else
    echo -e "${BOLD}${RED}Some tests failed.${NC} Please review errors above and:"
    echo "  • Run setup script: ./scripts/setup.sh"
    echo "  • Check Ollama: ollama serve"
    echo "  • Install dependencies: uv sync"
    echo "  • Pull models: ollama pull qwen3:8b"
    exit 1
fi
