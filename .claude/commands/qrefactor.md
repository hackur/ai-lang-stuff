---
description: Quick refactor following SOLID/DRY/KISS
---

## Usage
`/qrefactor <FILE_OR_CODE_DESCRIPTION>`

## Context
- Target: $ARGUMENTS
- Use @file to reference specific code

## Role
Refactor Coordinator with:
1. **Analyzer** – finds code smells
2. **Designer** – proposes SOLID/DRY/KISS improvements
3. **Implementer** – shows refactored code

## Process
1. Analyze current code
2. Identify violations (SOLID/DRY/KISS)
3. Design improvement
4. Show refactored code

## Output
- **Issues**: Code smells found
- **Design**: Refactoring approach
- **Code**: Refactored implementation
- **Impact**: Files changed, benefits gained
