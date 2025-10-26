# Custom Commands for PCR Card

Custom slash commands for the PCR Card Laravel project.

## Quick Commands (Fast & Focused)

### `/qfix` - Quick Fix
Rapidly debug and fix errors or issues.
```
/qfix TypeError in PaymentService line 42
/qfix Seeder failing with null constraint violation
```

### `/qrefactor` - Quick Refactor
Refactor code following SOLID/DRY/KISS principles.
```
/qrefactor PaymentService
/qrefactor @app/Services/StateTransitionService.php
```

### `/qtest` - Quick Test
Create tests or run test suites.
```
/qtest PaymentService
/qtest run unit
/qtest failing
```

### `/qnova` - Quick Nova Resource
Create Nova resources following project patterns.
```
/qnova Payment
/qnova Invoice user_id:belongsTo amount:currency
```

### `/qdb` - Quick Database
Database operations (migrations, seeders, queries).
```
/qdb migration add_status_to_users
/qdb seeder UserSeeder
/qdb fresh
```

### `/qsearch` - Quick Search
Search codebase with context and categorization.
```
/qsearch submitSubmission
/qsearch "old terminology"
/qsearch class PromoCode
```

### `/qdocs` - Quick Documentation
Update or create documentation.
```
/qdocs update
/qdocs PaymentService
/qdocs check
```

### `/qplan` - Quick Planning
Plan features or refactoring with step-by-step approach.
```
/qplan add payment retry logic
/qplan refactor state machine
```

### `/qcommit` - Quick Commit Message
Generate conventional commit message from staged changes.
```
/qcommit
```

### `/qreview` - Quick Code Review
Review staged changes for quality and issues.
```
/qreview
```

### `/ultrathink` - Deep Multi-Agent Analysis
Complex task orchestration with multiple specialist agents.
```
/ultrathink implement payment reconciliation system
```

---

## Comprehensive Commands (Detailed Workflows)

### `/session-summary`
Create comprehensive session summary document.

### `/review-staged`
Comprehensive review of all staged git changes.

### `/update-docs`
Update all documentation after code changes.

### `/refactor-plan`
Create detailed refactoring plan following SOLID/DRY/KISS.

### `/fresh-seed`
Reset database and seed with verification.

### `/commit-msg`
Generate comprehensive commit message for staged changes.

### `/code-quality`
Comprehensive code quality check (SOLID/DRY/KISS analysis).

### `/search-pattern`
Search for code patterns, terminology, or references.

### `/test-suite`
Run comprehensive test suite with reporting.

### `/nova-resource`
Create Laravel Nova resource following PCR Card best practices.

### `/quick-fix`
Quickly analyze and fix specific error or issue.

### `/migration-builder`
Create Laravel migration following project conventions.

### `/deploy-check`
Pre-deployment checklist and validation.

---

## Command Categories

### Development Workflow
- `/qfix` - Fast error fixing
- `/qrefactor` - Code improvement
- `/qplan` - Feature planning
- `/ultrathink` - Complex problem solving

### Database Operations
- `/qdb` - Database operations
- `/fresh-seed` - Database reset
- `/migration-builder` - Create migrations

### Nova Admin
- `/qnova` - Nova resource creation
- `/nova-resource` - Detailed Nova resource

### Testing
- `/qtest` - Test operations
- `/test-suite` - Full test suite

### Documentation
- `/qdocs` - Quick doc updates
- `/update-docs` - Comprehensive doc update
- `/session-summary` - Session documentation

### Code Analysis
- `/qsearch` - Codebase search
- `/qreview` - Code review
- `/code-quality` - Quality analysis
- `/search-pattern` - Pattern search

### Git & Deployment
- `/qcommit` - Commit message
- `/commit-msg` - Detailed commit message
- `/review-staged` - Review changes
- `/deploy-check` - Deployment validation

### Planning & Architecture
- `/qplan` - Quick planning
- `/refactor-plan` - Refactoring planning
- `/ultrathink` - Deep analysis

---

## Usage Tips

### Quick Commands (q*)
- Designed for speed and common tasks
- Use when you know exactly what you want
- Minimal ceremony, maximum action
- Example: `/qfix null error in PaymentService`

### Comprehensive Commands
- Use for complex, multi-step workflows
- Provide detailed analysis and documentation
- Include verification and validation
- Example: `/test-suite` for full test run with reporting

### UltraThink
- Use for complex, unclear problems
- Spawns multiple specialist agents
- Iterates until solution found
- Example: `/ultrathink design multi-tenant architecture`

### @file References
Most commands support @file syntax:
```
/qrefactor @app/Services/PaymentService.php
/qdocs @app/Models/Submission.php
```

---

## Quick Reference Card

**Most Common Commands:**
```bash
/qfix <error>              # Fix errors fast
/qrefactor <file>          # Improve code quality
/qtest <target>            # Test operations
/qdb fresh                 # Reset database
/qsearch <pattern>         # Find in codebase
/qcommit                   # Generate commit message
/qreview                   # Review staged changes
/qplan <task>              # Plan implementation
```

**Before Committing:**
```bash
/qreview                   # Check quality
/qcommit                   # Generate message
```

**Complex Problems:**
```bash
/ultrathink <description>  # Deep multi-agent analysis
```

---

## Creating New Commands

Add new .md files to `.claude/commands/`:

```markdown
---
description: Brief description shown in command list
---

## Usage
`/commandname <ARGUMENTS>`

## Context
- Argument: $ARGUMENTS

## Role
What this command does

## Process
1. Step 1
2. Step 2

## Output
- What the user gets
```

---

**Project**: PCR Card
**Documentation**: See individual command files for detailed usage
**Support**: Refer to CLAUDE.md for project-specific patterns
