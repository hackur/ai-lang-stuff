---
description: Quick planning for features or refactoring
---

## Usage
`/qplan <TASK>`

Examples:
- `/qplan add payment retry logic`
- `/qplan refactor state machine`
- `/qplan optimize submission queries`

## Context
- Task: $ARGUMENTS
- Use @file to reference affected code

## Role
Planning Coordinator with:
1. **Architect** – designs approach
2. **Estimator** – assesses complexity and effort
3. **Strategist** – orders implementation steps

## Process
1. Understand requirements and constraints
2. Design solution following SOLID/DRY/KISS
3. Break into small, testable steps
4. Estimate effort and identify risks

## Output
- **Approach**: High-level design
- **Steps**: 10-20 numbered implementation steps
- **Files**: What needs to change
- **Effort**: Time estimate
- **Risks**: Potential issues
