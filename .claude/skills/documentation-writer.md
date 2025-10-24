# Skill: Documentation Writer

## Purpose
Systematically create, update, and maintain project documentation to ensure clarity, completeness, and accessibility for all users from beginners to advanced developers.

## Triggers
- User asks to document a feature
- New code is written without documentation
- Existing documentation is outdated
- User mentions "docs", "documentation", "README"
- After implementing a new example or feature
- When troubleshooting reveals documentation gaps

## Process

### 1. Identify Documentation Need
Determine what needs to be documented:
- **New feature**: Complete usage guide
- **Bug fix**: Update troubleshooting section
- **API change**: Update relevant examples
- **Setup change**: Update installation instructions
- **Pattern discovery**: Add to best practices

### 2. Determine Documentation Location

**README.md** - User-facing, high-level:
- Quick start instructions
- Key features and benefits
- Common commands
- Basic troubleshooting
- Links to detailed docs

**plans/** - Detailed planning and architecture:
- Project vision and goals
- Research and findings
- Milestone roadmaps
- Comprehensive examples with full code

**CLAUDE.md** - Development assistant instructions:
- Development patterns
- Code quality standards
- Common tasks and solutions
- Troubleshooting workflows

**examples/** - Runnable code:
- Inline comments explaining why
- Docstrings with purpose, prerequisites, expected output
- Error handling with guidance

**plans/checklists/** - Step-by-step guides:
- Getting started
- Daily development workflow
- Milestone completion
- Deployment procedures

### 3. Write Documentation

Apply documentation standards:

**Structure:**
```markdown
# Title (Clear, Descriptive)

## Purpose/Overview
One paragraph explaining what and why.

## Prerequisites
- Required tools/knowledge
- Environment setup needed
- Dependencies

## Instructions
### Step 1: Clear Action
Explanation of what this does.

```bash
# Command to run
```

**Expected output:**
What the user should see.

### Step 2: Next Action
...

## Examples
Working code examples.

## Troubleshooting
Common issues and solutions.

## Next Steps
What to do after completing this.
```

**Code Documentation:**
```python
def function_name(param: type) -> return_type:
    """
    One-line summary of what this function does.

    More detailed explanation if needed, describing the approach
    and any important considerations.

    Args:
        param: Description of parameter and expected values

    Returns:
        Description of return value and format

    Raises:
        ErrorType: When and why this error occurs

    Example:
        >>> result = function_name("test")
        >>> print(result)
        Expected output
    """
```

**Commit Message Documentation:**
```
Type(scope): Brief summary (50 chars or less)

Detailed explanation of the change and why it was necessary.
Include context about the problem being solved.

What changed:
- Specific change 1
- Specific change 2

Why changed:
- Reason for change
- Impact on users/system

Breaking changes:
- Note any breaking changes

Testing:
- How this was tested
```

### 4. Ensure Completeness

Documentation checklist:
- [ ] Purpose/goal clearly stated
- [ ] Prerequisites listed
- [ ] Step-by-step instructions
- [ ] Expected outputs described
- [ ] Code examples included and tested
- [ ] Common errors documented
- [ ] Troubleshooting section
- [ ] Next steps provided
- [ ] Links to related documentation
- [ ] Diagrams if helpful
- [ ] Up to date with current code

### 5. Validate Documentation

Test documentation by following it:
- [ ] Can a beginner follow these instructions?
- [ ] Are all commands correct and up to date?
- [ ] Do all code examples work?
- [ ] Are prerequisites complete?
- [ ] Is troubleshooting section helpful?
- [ ] Are there any assumptions not stated?
- [ ] Is the language clear and concise?

### 6. Maintain Documentation

Keep documentation current:
- Update when code changes
- Add new troubleshooting as issues arise
- Incorporate user feedback
- Remove outdated information
- Cross-reference related docs
- Verify links still work

## Documentation Types

### Tutorial Documentation
- Step-by-step guide for specific task
- Assumes beginner knowledge
- Includes all necessary context
- Shows expected output at each step
- Example: Getting Started Checklist

### Reference Documentation
- Complete API/function reference
- Organized by category
- Search-friendly
- Technical and precise
- Example: CLAUDE.md

### Explanation Documentation
- Explains concepts and decisions
- Provides context and rationale
- Shows relationships between components
- Example: plans/0-readme.md (vision)

### How-To Documentation
- Solves specific problem
- Assumes some knowledge
- Focuses on one task
- Practical and actionable
- Example: Daily Development Checklist

## Output

### For New Features
Create or update:
- README.md with high-level usage
- Relevant plan document with detailed example
- Example file with working code
- Test file validating functionality
- IMPLEMENTATION_SUMMARY.md noting change

### For Bug Fixes
Update:
- Troubleshooting section in relevant docs
- Add to common issues list
- Note resolution in commit message

### For API Changes
Update:
- All affected examples
- API reference in CLAUDE.md
- Migration guide if breaking change
- Deprecation warnings if applicable

## Example Interaction

**User**: "I just added a new RAG example. Can you document it?"

**Response**:
"I'll document your new RAG example comprehensively. Let me:

1. Add detailed documentation to the example file:
   - Purpose and use case
   - Prerequisites
   - Step-by-step code explanation
   - Expected output
   - Troubleshooting tips

2. Update plans/3-kitchen-sink-plan.md:
   - Add to Example 4 section
   - Include complete code
   - Add checklist for validation

3. Update README.md:
   - Add to usage examples
   - Include in learning path

4. Update CLAUDE.md:
   - Add RAG pattern to development guidelines
   - Note best practices discovered

5. Create test case in tests/:
   - Validate RAG functionality
   - Test vector store integration
   - Check retrieval quality

[Proceeds to create documentation following the structure above]

Documentation complete! The RAG example is now:
- Fully documented with inline comments
- Included in comprehensive example guide
- Referenced in README
- Covered by tests
- Development patterns documented

Users can now:
- Understand what the example does
- Follow step-by-step to implement RAG
- Troubleshoot common issues
- Extend for their use cases"

## Best Practices

1. **Write for your audience**: Beginners need more context, advanced users need reference
2. **Show, don't just tell**: Code examples are worth 1000 words
3. **Test your documentation**: Actually follow the steps you wrote
4. **Keep it current**: Update docs when code changes
5. **Be concise but complete**: Every necessary detail, no unnecessary words
6. **Use consistent structure**: Readers know where to find information
7. **Include visuals when helpful**: Diagrams, screenshots, output examples
8. **Link liberally**: Connect related documentation
9. **Admit limitations**: Document known issues and workarounds
10. **Make it searchable**: Use clear headings and keywords

## Documentation Metrics

Good documentation when:
- User can complete task without asking questions
- Common issues have documented solutions
- Code examples all work as shown
- No undocumented prerequisites
- Clear next steps provided
- Searchable and well-organized
- Maintained and current
