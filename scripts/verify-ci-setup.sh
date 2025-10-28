#!/bin/bash
# Verify CI/CD setup is complete and functional

set -e

echo "🔍 Verifying CI/CD Setup..."
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check counter
CHECKS_PASSED=0
CHECKS_FAILED=0

check() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $1"
        ((CHECKS_PASSED++))
    else
        echo -e "${RED}✗${NC} $1"
        ((CHECKS_FAILED++))
    fi
}

# Check GitHub workflows
echo "📋 Checking GitHub Workflows..."
test -f .github/workflows/test.yml && check "Test workflow exists"
test -f .github/workflows/lint.yml && check "Lint workflow exists"
test -f .github/workflows/docs.yml && check "Docs workflow exists"
test -f .github/workflows/benchmarks.yml && check "Benchmarks workflow exists"
echo ""

# Check GitHub templates
echo "📝 Checking GitHub Templates..."
test -f .github/PULL_REQUEST_TEMPLATE.md && check "PR template exists"
test -f .github/ISSUE_TEMPLATE/bug_report.md && check "Bug report template exists"
test -f .github/ISSUE_TEMPLATE/feature_request.md && check "Feature request template exists"
echo ""

# Check root configuration files
echo "⚙️  Checking Configuration Files..."
test -f Makefile && check "Makefile exists"
test -f .pre-commit-config.yaml && check "Pre-commit config exists"
test -f mkdocs.yml && check "MkDocs config exists"
echo ""

# Check pyproject.toml configuration
echo "📦 Checking pyproject.toml Configuration..."
grep -q "pytest-benchmark" pyproject.toml && check "pytest-benchmark dependency"
grep -q "pre-commit" pyproject.toml && check "pre-commit dependency"
grep -q "mkdocs" pyproject.toml && check "mkdocs dependency"
grep -q "\[tool.bandit\]" pyproject.toml && check "Bandit configuration"
grep -q "\[tool.coverage" pyproject.toml && check "Coverage configuration"
echo ""

# Check documentation
echo "📚 Checking Documentation..."
test -f docs/CI-CD.md && check "CI/CD documentation exists"
test -f .github/workflows/README.md && check "Workflows README exists"
test -f .github/CI-CD-QUICK-START.md && check "Quick start guide exists"
echo ""

# Check benchmark infrastructure
echo "🏃 Checking Benchmark Infrastructure..."
test -d tests/benchmarks && check "Benchmarks directory exists"
test -f tests/benchmarks/test_llm_performance.py && check "Benchmark tests exist"
test -f scripts/compare_benchmarks.py && check "Benchmark comparison script exists"
test -x scripts/compare_benchmarks.py && check "Benchmark script is executable"
echo ""

# Check Makefile targets
echo "🔨 Checking Makefile Targets..."
grep -q "^test:" Makefile && check "test target exists"
grep -q "^lint:" Makefile && check "lint target exists"
grep -q "^format:" Makefile && check "format target exists"
grep -q "^benchmark:" Makefile && check "benchmark target exists"
grep -q "^docs:" Makefile && check "docs target exists"
grep -q "^ci:" Makefile && check "ci target exists"
echo ""

# Check pre-commit hooks configuration
echo "🪝 Checking Pre-commit Hooks..."
grep -q "ruff-pre-commit" .pre-commit-config.yaml && check "Ruff hook configured"
grep -q "black" .pre-commit-config.yaml && check "Black hook configured"
grep -q "mypy" .pre-commit-config.yaml && check "mypy hook configured"
grep -q "bandit" .pre-commit-config.yaml && check "Bandit hook configured"
echo ""

# Test Makefile commands
echo "🧪 Testing Makefile Commands..."
make help > /dev/null 2>&1 && check "Makefile help works"
echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
if [ $CHECKS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed!${NC} ($CHECKS_PASSED/$((CHECKS_PASSED + CHECKS_FAILED)))"
    echo ""
    echo "🚀 CI/CD setup is complete and ready to use!"
    echo ""
    echo "Next steps:"
    echo "  1. Install pre-commit hooks: make setup"
    echo "  2. Run CI checks locally: make ci"
    echo "  3. View documentation: make docs-serve"
    echo ""
else
    echo -e "${YELLOW}⚠️  Some checks failed${NC} ($CHECKS_FAILED failed, $CHECKS_PASSED passed)"
    echo ""
    echo "Please review the failed checks above."
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

exit $CHECKS_FAILED
