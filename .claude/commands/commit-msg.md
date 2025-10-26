---
description: Generate a comprehensive commit message for staged changes
---

Generate a comprehensive, standards-compliant commit message for all staged changes:

## Commit Message Format

Follow Conventional Commits with PCR Card enhancements:

```
<type>(<scope>): <short description>

<detailed description>

Changes:
- <bullet point list of changes>
- <organized by file/component>
- <specific and actionable>

Benefits:
- <why this change improves the codebase>
- <impact on users/developers>

Technical Details:
- <implementation specifics>
- <patterns applied (SOLID/DRY/KISS)>
- <performance/quality metrics>

Breaking Changes: <if any, list them>
Migration: <if needed, provide steps>

Files changed: <count>
Lines modified: <+additions -deletions>
Tests: <status>
Documentation: <status>
```

## Types
- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation only
- **refactor**: Code refactoring
- **perf**: Performance improvement
- **test**: Adding/updating tests
- **chore**: Maintenance tasks

## Scopes (PCR Card specific)
- seeders, database, migrations
- nova, admin, resources
- api, endpoints, controllers
- services, models, repositories
- tests, dusk, phpunit
- scripts, deployment, ci-cd
- docs, guides, summaries

## Analysis Steps

1. Run `git diff --staged --stat` for file summary
2. Run `git diff --staged` to see all changes
3. Identify the primary type and scope
4. List all changes by component
5. Explain the benefits and reasoning
6. Note any breaking changes or migrations needed
7. Include metrics (files, lines, test status)

Generate the commit message following this format.
