---
description: Quick documentation update
---

## Usage
`/qdocs <TARGET>`

Examples:
- `/qdocs update` - Update docs for recent changes
- `/qdocs PaymentService` - Document PaymentService class
- `/qdocs check` - Find outdated documentation

## Context
- Target: $ARGUMENTS

## Role
Doc Coordinator with:
1. **Auditor** – finds what needs documenting
2. **Writer** – creates/updates documentation
3. **Validator** – ensures accuracy

## Process
1. Identify documentation needs
2. Update or create docs
3. Verify examples work
4. Check cross-references

## Output
- **Updated**: Files modified
- **Added**: New documentation
- **Verified**: Examples tested
- **Next**: Additional docs needed
