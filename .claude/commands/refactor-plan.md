---
description: Create a detailed refactoring plan following SOLID/DRY/KISS
---

Create a comprehensive refactoring plan for the specified code:

## Refactoring Analysis

1. **Current State Assessment**
   - Read the current code implementation
   - Identify code smells and anti-patterns
   - Measure current metrics (lines, complexity, duplication)
   - Document current behavior

2. **SOLID Principles Violations**
   - **Single Responsibility**: List classes/methods doing too much
   - **Open/Closed**: Identify hard-to-extend code
   - **Liskov Substitution**: Check inheritance issues
   - **Interface Segregation**: Find fat interfaces
   - **Dependency Inversion**: Locate tight coupling

3. **DRY Violations**
   - Find duplicated code blocks
   - Identify repeated patterns
   - Locate copy-paste code
   - Calculate duplication percentage

4. **KISS Violations**
   - Find overly complex methods
   - Identify unnecessary abstractions
   - Locate convoluted logic
   - Find premature optimizations

5. **Refactoring Strategy**
   - Prioritize changes by impact
   - Break into phases (small, iterative changes)
   - Identify which patterns to apply (Strategy, Factory, etc.)
   - Plan for backward compatibility

6. **Implementation Plan**
   - Create 20+ step detailed plan
   - Specify which files to modify
   - Define new classes/methods needed
   - Plan for testing each change

7. **Risk Assessment**
   - Identify breaking changes
   - List affected code areas
   - Plan migration strategy
   - Define rollback plan

8. **Success Metrics**
   - Target metrics (lines reduced, complexity lowered)
   - Code quality improvements
   - Performance expectations
   - Maintainability gains

Execute the refactoring following the plan, or present plan for approval first.
