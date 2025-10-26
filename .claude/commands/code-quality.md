---
description: Comprehensive code quality check (SOLID/DRY/KISS analysis)
---

Perform a comprehensive code quality analysis on specified files or recent changes:

## Code Quality Analysis

### 1. SOLID Principles Check

**Single Responsibility Principle (SRP)**
- Identify classes/methods doing multiple things
- Check for mixed concerns (business logic + presentation)
- Look for "God classes" or "God methods"
- Measure method length (target: <20 lines)

**Open/Closed Principle (OCP)**
- Find hardcoded values that should be configurable
- Identify switch statements that should be polymorphic
- Look for modification-prone code

**Liskov Substitution Principle (LSP)**
- Check inheritance hierarchies
- Verify subtypes don't break parent contracts
- Look for instanceof checks (code smell)

**Interface Segregation Principle (ISP)**
- Find "fat interfaces" with too many methods
- Check if clients depend on unused methods
- Look for interface pollution

**Dependency Inversion Principle (DIP)**
- Check for tight coupling
- Look for direct instantiation (should use injection)
- Verify high-level modules don't depend on low-level details

### 2. DRY (Don't Repeat Yourself)

- Search for duplicate code blocks
- Identify copy-pasted methods
- Find repeated patterns
- Calculate duplication percentage
- Suggest extraction opportunities

### 3. KISS (Keep It Simple, Stupid)

- Identify overly complex methods (cyclomatic complexity >10)
- Find unnecessary abstractions
- Look for premature optimizations
- Check for convoluted logic
- Suggest simplification opportunities

### 4. Code Smells

- Long methods (>50 lines)
- Large classes (>500 lines)
- Long parameter lists (>4 params)
- Magic numbers and strings
- Commented-out code
- TODO/FIXME comments

### 5. Laravel/PHP Best Practices

- Check for proper Eloquent usage
- Verify service layer pattern
- Check repository pattern compliance
- Look for N+1 query issues
- Verify proper exception handling

### 6. Documentation Quality

- PHPDoc coverage percentage
- Missing @param or @return tags
- Outdated documentation
- Missing inline comments on complex logic

### 7. Metrics Summary

Provide overall scores:
- SOLID Compliance: X/10
- DRY Score: X/10 (100% - duplication%)
- KISS Score: X/10 (based on average complexity)
- Documentation Coverage: X%
- Overall Code Quality Grade: A-F

### 8. Recommendations

- Prioritized list of improvements
- Specific refactoring suggestions
- Reference to patterns to apply
- Estimated effort for each fix

Output a detailed report with specific line numbers and actionable recommendations.
