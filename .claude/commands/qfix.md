---
description: Quick fix for errors or issues
---

## Usage
`/qfix <ERROR_OR_ISSUE>`

## Context
- Issue: $ARGUMENTS
- Use @file syntax to reference specific files

## Role
Debug Coordinator orchestrating:
1. **Analyzer** – identifies root cause
2. **Fixer** – provides solution
3. **Validator** – suggests verification

## Process
1. Read error context
2. Identify root cause
3. Provide immediate fix
4. Suggest verification

## Output
- **Root Cause**: One-line explanation
- **Fix**: Code or command to execute
- **Verify**: How to confirm it works
- **Prevent**: Pattern to avoid this in future
