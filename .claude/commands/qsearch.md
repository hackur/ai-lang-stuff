---
description: Quick codebase search with context
---

## Usage
`/qsearch <PATTERN>`

Examples:
- `/qsearch submitSubmission` - Find method usage
- `/qsearch "old term"` - Find terminology to update
- `/qsearch class PromoCode` - Find class definition

## Context
- Pattern: $ARGUMENTS

## Role
Search Coordinator with:
1. **Searcher** – finds all matches using Grep/Glob
2. **Analyzer** – provides context and categorization
3. **Reporter** – summarizes findings

## Process
1. Search codebase with appropriate tools
2. Categorize results (code, docs, tests, scripts)
3. Provide context for each match
4. Suggest actions

## Output
- **Matches**: Count and categories
- **Results**: Grouped by file type with line numbers
- **Context**: Code surrounding matches
- **Actions**: What to do with findings
