# Skill: Git Commit Organizer

## Purpose
Systematically organize large sets of unstaged or poorly organized changes into logical, well-documented commit groups with comprehensive messages. This skill helps maintain clean git history and makes code reviews easier.

## When to Use
- Large number of files changed (50+ files)
- Mixed changes across multiple features/systems
- Repository cleanup or refactoring
- After bulk code generation
- Before major PR submission
- When git history is messy

## Prerequisites
- Git repository initialized
- Changes present (staged or unstaged)
- Understanding of the codebase structure

## Process

### 1. **Analysis Phase**
```bash
# Check current state
git status --short
git diff --stat
git log --oneline -10

# Identify file groups
git status --short | cut -c4- | cut -d/ -f1 | sort | uniq -c
```

**Analyze:**
- Total files changed
- Logical groupings (by directory, feature, type)
- Dependencies between changes
- Potential commit boundaries

### 2. **Planning Phase**
Create logical commit groups based on:
- **Functionality**: Features, fixes, refactors
- **Scope**: Core vs supporting vs documentation
- **Dependencies**: What must go together
- **Review**: Easier to review in chunks

**Example Grouping:**
```
Group 1: Core utilities (utils/*.py + tests)
Group 2: Documentation (docs/, README.md)
Group 3: Examples (examples/01-*, examples/02-*)
Group 4: CI/CD infrastructure (.github/*)
Group 5: Configuration (pyproject.toml, .gitignore)
```

### 3. **Execution Phase**

For each group:

```bash
# Unstage everything first
git reset HEAD

# Stage specific group
git add utils/ tests/test_*

# Verify what's staged
git diff --cached --stat

# Commit with comprehensive message
git commit -m "$(cat <<'EOF'
[Title: Clear, imperative, <50 chars]

[Body: Detailed explanation]
- What changed
- Why it changed
- Impact and dependencies

**Files Changed:**
- file1.py: description
- file2.py: description

**Testing:**
- How to test
- Expected results

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

### 4. **Verification Phase**

```bash
# Check commit history
git log --oneline -10

# Verify no __pycache__ or binaries
git log --all --name-only | grep -E '(__pycache__|\.pyc$|\.DS_Store)'

# Check commit sizes
git log --oneline --shortstat -10
```

### 5. **Cleanup Phase**

If mistakes found:
```bash
# Amend last commit (if not pushed)
git add forgotten-file.py
git commit --amend --no-edit

# Reorder commits (if not pushed)
git rebase -i HEAD~5

# Remove unwanted files from history
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch path/to/file' \
  --prune-empty HEAD
```

## Commit Message Template

```markdown
[Type]([Scope]): [Subject - imperative, <50 chars]

[Detailed description of what and why, not how]

**Changes:**
- Component 1: What changed and why
- Component 2: What changed and why
- Component 3: What changed and why

**Files Added:** (if applicable)
1. file1.py (lines) - Purpose
2. file2.py (lines) - Purpose

**Files Modified:** (if applicable)
- file3.py: What was modified

**Testing:**
- How to test these changes
- Expected behavior

**Dependencies:** (if applicable)
- Requires: Other commits/branches
- Blocks: Future work

**Breaking Changes:** (if applicable)
- What breaks
- Migration path

[Footer: Co-author, issue refs, etc.]
🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

### Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, no code change
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance
- `perf`: Performance improvement

## Best Practices

### DO:
✅ Group related changes together
✅ Write detailed commit messages
✅ Explain "why", not just "what"
✅ Keep commits focused and atomic
✅ Review each commit before finalizing
✅ Use consistent formatting
✅ Add co-author attribution
✅ Reference issues/PRs when relevant

### DON'T:
❌ Mix unrelated changes
❌ Create huge commits (>1000 lines unless necessary)
❌ Write vague messages ("fixed stuff", "updates")
❌ Commit generated files (__pycache__, .DS_Store)
❌ Commit secrets or credentials
❌ Rewrite published history
❌ Use ambiguous pronouns in messages

## Real-World Example

**Scenario:** 161 files changed, 66,443 insertions

**Solution:**
```bash
# 1. Unstage all
git reset HEAD

# 2. Create 12 logical groups
git add utils/ tests/test_*.py
git commit -m "Add core utility modules and tests (7,170 lines)"

git add docs/
git commit -m "Add comprehensive documentation (18,804 lines)"

# ... 10 more commits

# 3. Verify
git log --oneline -12
# Shows clean, logical progression

# 4. Push
git push origin feature-branch
```

## Common Pitfalls

### Problem: Too Many Files Staged
```bash
# Solution: Use selective staging
git add -p  # Interactive patch mode
git add specific/files/only
```

### Problem: Committed Wrong Files
```bash
# Solution: Amend or reset
git reset HEAD~1  # Undo last commit, keep changes
git commit -m "Correct commit"
```

### Problem: Binary Files in History
```bash
# Solution: Filter branch
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch *.pyc' \
  --prune-empty HEAD
```

## Output Format

After completion, provide:

1. **Summary Statistics:**
   - Total commits created
   - Files per commit
   - Lines changed per commit

2. **Commit List:**
   - Hash and title for each
   - Brief description

3. **Next Steps:**
   - Push command
   - PR creation if applicable
   - Testing recommendations

## Integration with Other Tools

```bash
# Pre-commit hooks
pre-commit install
pre-commit run --all-files

# Commit message validation
npm install -g @commitlint/cli
commitlint --from HEAD~1 --to HEAD

# Conventional commits
npm install -g commitizen
git cz
```

## Success Criteria

- ✅ All commits build successfully
- ✅ Each commit is logically coherent
- ✅ Commit messages are clear and detailed
- ✅ No binary/generated files committed
- ✅ Git history is linear and clean
- ✅ Easy to review and cherry-pick

## Advanced: Scripted Organization

```bash
#!/bin/bash
# organize-commits.sh

# Get all modified files
files=$(git diff --name-only)

# Group by directory
for dir in $(echo "$files" | cut -d/ -f1 | sort -u); do
  echo "Processing $dir..."
  git add "$dir/"
  git commit -m "Update $dir/ directory"
done
```

## Related Skills
- `git-history-rewriter` - For rewriting git history
- `commit-message-validator` - For validating messages
- `pr-preparation` - For preparing pull requests

---

**Skill Level:** Intermediate to Advanced
**Time:** 30 minutes - 2 hours depending on complexity
**Tools:** git, text editor, terminal
