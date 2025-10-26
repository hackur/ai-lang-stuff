---
description: Quick commit message generation
---

## Usage
`/qcommit`

Analyzes staged changes and generates a commit message.

## Role
Commit Coordinator with:
1. **Analyzer** – reviews staged changes
2. **Categorizer** – determines type and scope
3. **Writer** – crafts conventional commit message

## Process
1. Run `git diff --staged --stat`
2. Identify primary type (feat/fix/refactor/docs/etc)
3. Determine scope (seeders/nova/api/etc)
4. Write commit message following conventions

## Output
```
<type>(<scope>): <description>

<detailed explanation>

Changes:
- <specific changes>

Files: X changed, +Y -Z
Breaking: <if any>
```

Ready to copy and use with `git commit -m "..."`
