#!/bin/bash
#
# Test LangGraph Studio Setup
# Verifies all components are properly configured
#

set -e

echo "======================================================================"
echo "LangGraph Studio Setup Verification"
echo "======================================================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track errors
ERRORS=0

# Test function
test_item() {
    local name=$1
    local cmd=$2

    echo -n "Testing $name... "
    if eval "$cmd" &>/dev/null; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
        ((ERRORS++))
    fi
}

test_file() {
    local name=$1
    local file=$2

    echo -n "Checking $name... "
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗ (missing: $file)${NC}"
        ((ERRORS++))
    fi
}

test_directory() {
    local name=$1
    local dir=$2

    echo -n "Checking $name... "
    if [ -d "$dir" ]; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗ (missing: $dir)${NC}"
        ((ERRORS++))
    fi
}

# 1. Prerequisites
echo "1. Prerequisites"
echo "----------------------------------------------------------------------"
test_item "Node.js 18+" "node --version | grep -E 'v(1[8-9]|[2-9][0-9])'"
test_item "Python 3.10+" "python3 --version | grep -E 'Python 3\.(1[0-9]|[2-9][0-9])'"
test_item "Ollama installed" "which ollama"
echo ""

# 2. Configuration Files
echo "2. Configuration Files"
echo "----------------------------------------------------------------------"
test_file "langgraph.json" "./langgraph.json"
test_file ".env.example" "./.env.example"
test_file "package.json" "./package.json"
echo ""

# 3. Workflow Files
echo "3. Workflow Files"
echo "----------------------------------------------------------------------"
test_directory "workflows directory" "./workflows"
test_file "research_agent.py" "./workflows/research_agent.py"
test_file "code_reviewer.py" "./workflows/code_reviewer.py"
test_file "rag_pipeline.py" "./workflows/rag_pipeline.py"
test_file "workflows __init__.py" "./workflows/__init__.py"
echo ""

# 4. Documentation
echo "4. Documentation"
echo "----------------------------------------------------------------------"
test_file "Studio guide" "./docs/langgraph-studio-guide.md"
test_file "Workflows README" "./workflows/README.md"
test_file "Quick start" "./LANGGRAPH-STUDIO-QUICKSTART.md"
echo ""

# 5. Directories
echo "5. Required Directories"
echo "----------------------------------------------------------------------"
test_directory "checkpoints" "./checkpoints"
test_directory "data/chroma" "./data/chroma"
echo ""

# 6. Workflow Syntax
echo "6. Workflow Syntax Check"
echo "----------------------------------------------------------------------"
echo -n "Checking research_agent.py syntax... "
if python3 -c "import ast; ast.parse(open('workflows/research_agent.py').read())" 2>/dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    ((ERRORS++))
fi

echo -n "Checking code_reviewer.py syntax... "
if python3 -c "import ast; ast.parse(open('workflows/code_reviewer.py').read())" 2>/dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    ((ERRORS++))
fi

echo -n "Checking rag_pipeline.py syntax... "
if python3 -c "import ast; ast.parse(open('workflows/rag_pipeline.py').read())" 2>/dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    ((ERRORS++))
fi
echo ""

# 7. JSON Validation
echo "7. JSON Configuration Validation"
echo "----------------------------------------------------------------------"
echo -n "Validating langgraph.json... "
if python3 -m json.tool langgraph.json >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    ((ERRORS++))
fi

echo -n "Validating package.json... "
if python3 -m json.tool package.json >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    ((ERRORS++))
fi
echo ""

# 8. Ollama Status
echo "8. Ollama Status"
echo "----------------------------------------------------------------------"
echo -n "Checking if Ollama is running... "
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"

    echo -n "Checking for qwen3:8b model... "
    if ollama list | grep -q "qwen3:8b"; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${YELLOW}⚠ (model not installed)${NC}"
        echo "  Run: ollama pull qwen3:8b"
    fi
else
    echo -e "${YELLOW}⚠ (not running)${NC}"
    echo "  Run: ollama serve"
fi
echo ""

# 9. NPM Scripts
echo "9. NPM Scripts"
echo "----------------------------------------------------------------------"
echo -n "Checking 'npm run studio' script... "
if grep -q '"studio".*langgraph.*dev' package.json; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    ((ERRORS++))
fi
echo ""

# Summary
echo "======================================================================"
echo "Summary"
echo "======================================================================"

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}All checks passed!${NC}"
    echo ""
    echo "You're ready to use LangGraph Studio:"
    echo "  1. Start Ollama (if not running): ollama serve"
    echo "  2. Pull model (if needed): ollama pull qwen3:8b"
    echo "  3. Start Studio: npm run studio"
    echo "  4. Open: http://localhost:8123/studio"
    echo ""
    exit 0
else
    echo -e "${RED}Found $ERRORS error(s)${NC}"
    echo ""
    echo "Please fix the errors above before running Studio."
    echo ""
    exit 1
fi
