---
description: Search for code patterns, terminology, or references across the project
---

Search comprehensively for code patterns, terminology, or references:

## Search Strategy

### 1. Pattern Identification
- What are you searching for? (pattern, term, reference)
- Where to search? (specific dirs or entire project)
- File types? (*.php, *.md, *.js, etc.)

### 2. Multi-Tool Search Approach

**Use Grep for**:
- Code patterns (`grep -r "pattern" --include="*.php"`)
- Class/method references
- String literals
- Import statements

**Use Glob for**:
- File name patterns (`**/*Seeder.php`)
- Directory structures
- File organization

**Use Read for**:
- Examining specific files found
- Verifying context
- Detailed analysis

### 3. Search Patterns to Check

**For Terminology Updates**:
- Old term (case-insensitive)
- Variations (plural, past tense, etc.)
- Comments and documentation
- Variable/method names
- String literals

**For Refactoring**:
- All usages of class/method
- Import statements
- Test references
- Documentation mentions

**For Deprecation**:
- Find all usage sites
- Check for replacement availability
- Identify migration path

### 4. Organized Results

Present findings grouped by:
- **Code Files**: Actual usage in PHP/JS
- **Documentation**: References in .md files
- **Tests**: Usage in test files
- **Configuration**: Config/environment files
- **Scripts**: Shell script references

### 5. Impact Analysis

For each finding:
- File path and line number
- Context (surrounding code)
- Impact (critical, moderate, low)
- Suggested action (update, remove, migrate)

### 6. Summary Report

- Total matches found
- Files affected
- Breakdown by category
- Priority order for updates
- Estimated effort

Provide a comprehensive search report with specific line numbers and context.
