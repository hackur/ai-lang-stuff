---
description: Quick test creation or test suite run
---

## Usage
`/qtest <TEST_TARGET>`

Examples:
- `/qtest PaymentService` - Create tests for PaymentService
- `/qtest run unit` - Run unit test suite
- `/qtest failing` - Debug failing tests

## Context
- Target: $ARGUMENTS

## Role
Test Coordinator with:
1. **Planner** – determines test strategy
2. **Writer** – creates test code
3. **Runner** – executes and reports

## Process
1. Understand what needs testing
2. Create or run tests
3. Report results
4. Fix failures if needed

## Output
- **Tests**: Test code or execution results
- **Coverage**: What's tested
- **Status**: Pass/fail summary
- **Next**: Actions if failures
