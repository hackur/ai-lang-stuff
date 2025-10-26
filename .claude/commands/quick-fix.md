---
description: Quickly analyze and fix a specific error or issue
---

Analyze and fix a specific error or issue:

## Quick Fix Process

### 1. Error Analysis

Provide the error:
- Full error message
- Stack trace
- File and line number
- Context (what were you doing?)

### 2. Immediate Investigation

I will:
- Read the failing file at the specified line
- Check surrounding context
- Look for obvious issues
- Search for similar patterns in codebase

### 3. Common Laravel/Nova Issues

**Check for these common problems**:

**Database/Eloquent**:
- Missing fillable/guarded
- Incorrect relationship definitions
- N+1 query problems
- Migration not run

**Nova**:
- Badge field using ::make instead of closure
- DateTime field with ->asHtml() (doesn't exist)
- Missing searchableColumns() method
- Incorrect field types

**State Machine**:
- Using lowercase state strings instead of State classes
- Incorrect transition permissions
- Missing state constants

**Testing**:
- Missing database setup
- Incorrect assertions
- Missing dependencies

### 4. Root Cause Identification

Determine:
- What caused the error?
- Why did it happen?
- Is it a code bug or environment issue?
- Are there related issues?

### 5. Fix Implementation

- Provide exact fix needed
- Show before/after code
- Explain why fix works
- Check for similar issues elsewhere

### 6. Verification

- What tests to run?
- How to manually verify?
- What to watch for?

### 7. Prevention

- How to prevent this in future?
- What pattern to follow?
- Any documentation to update?

Provide immediate fix and explanation.
