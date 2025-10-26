---
description: Run comprehensive test suite with reporting
---

Run the complete test suite with comprehensive reporting:

## Test Execution Plan

### 1. Pre-Test Preparation
- Verify Sail is running: `./vendor/bin/sail ps`
- Check database connectivity
- Ensure test database is ready
- Clear test caches

### 2. Unit Tests
```bash
./vendor/bin/sail artisan test --testsuite=Unit
```
- Run all unit tests
- Report pass/fail count
- List any failures with details
- Show execution time

### 3. Feature Tests
```bash
./vendor/bin/sail artisan test --testsuite=Feature
```
- Run all feature tests
- Report pass/fail count
- Identify slow tests (>1s)
- List failures

### 4. Browser Tests (Dusk)
```bash
./vendor/bin/sail dusk
```
- Run Dusk browser tests
- Check for screenshot generation on failures
- Report browser compatibility
- List any failures

### 5. Code Coverage (Optional)
```bash
./vendor/bin/sail artisan test --coverage --min=70
```
- Generate coverage report
- Identify uncovered code
- Check if minimum threshold met

### 6. Static Analysis (Optional)
```bash
./vendor/bin/sail composer phpstan
./vendor/bin/sail composer pint --test
```
- Run PHPStan analysis
- Check code style with Laravel Pint
- Report any violations

### 7. Test Summary Report

Provide comprehensive summary:
- **Total Tests**: X passed, Y failed, Z skipped
- **Execution Time**: Total and per suite
- **Coverage**: X% (if run)
- **Failures**: Detailed list with file/line
- **Slow Tests**: List tests >1s
- **Recommendations**: Actions to fix failures

### 8. Failure Analysis

For each failure:
- Test name and file
- Expected vs actual
- Stack trace (first 10 lines)
- Suggested fix
- Related code to check

### 9. Next Steps

If failures exist:
- Prioritize fixes by impact
- Suggest investigation areas
- Link to relevant documentation
- Provide debugging commands

Report final status: [PASS] All Passing or [FAIL] X Failures Found
