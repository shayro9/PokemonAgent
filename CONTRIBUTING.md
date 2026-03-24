# Contributing to PokemonAgent

Thank you for contributing! This guide ensures we maintain a clean, organized repository with clear history.

## 🌿 Branching Strategy: Trunk-Based Development

We use **trunk-based development** with short-lived feature branches. `main` is always production-ready and deployable.

### Branch Types & Naming

| Type | Pattern | Example | Purpose |
|------|---------|---------|---------|
| Feature | `feat/*` | `feat/lstm-improvements` | New features or enhancements |
| Bug Fix | `fix/*` | `fix/negative-reward-bug` | Bug fixes |
| Chore | `chore/*` | `chore/deps-update` | Dependencies, tooling, cleanup |
| Docs | `docs/*` | `docs/training-guide` | Documentation only |
| Refactor | `refactor/*` | `refactor/combat-logic` | Code improvements (no behavior change) |
| Test | `test/*` | `test/battle-integration` | Test additions/fixes |

### Starting Your Work

1. **Create a branch from `main`:**
   ```bash
   git fetch origin
   git checkout -b feat/my-feature origin/main
   ```

2. **Or use worktrees for parallel work:**
   ```bash
   git worktree add ../my-feature feat/my-feature
   cd ../my-feature
   ```

3. **Keep your branch updated:**
   ```bash
   git fetch origin
   git rebase origin/main
   ```

## 📝 Commit Messages (Conventional Commits)

We follow [Conventional Commits](https://www.conventionalcommits.org/) for clear history and automated tooling.

### Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Examples

**Simple fix:**
```
fix(agents): correct negative reward calculation

The reward system was inverting scores when team size > 1.
```

**Feature with details:**
```
feat(combat): implement 2v2 battle system

- Added team selection logic
- Implemented parallel action processing
- Fixed reward aggregation for teams
- Updated state representation for team tracking

Closes #42
```

**Chore:**
```
chore(deps): upgrade tensorflow to 2.14.0
```

### Type Definitions
- **feat**: New feature
- **fix**: Bug fix
- **chore**: Maintenance, deps, tooling
- **docs**: Documentation changes only
- **refactor**: Code improvement (no behavior change)
- **test**: Test additions or fixes
- **perf**: Performance improvement

## 🔄 Pull Request Workflow

### Before Creating a PR

1. **Clean up your commits:**
   ```bash
   git fetch origin
   git rebase -i origin/main
   ```
   - Squash fixup commits
   - Reword messages if needed
   - Keep commits atomic (one logical change per commit)

2. **Push to your branch:**
   ```bash
   git push -u origin feat/my-feature
   ```

3. **If you need to update after push:**
   ```bash
   git push --force-with-lease  # Safe force push (only you can do this)
   ```

### Creating the PR

1. Use the PR template
2. Link related issues: "Closes #42" or "Fixes #15"
3. Provide context: what changed and why
4. Request reviews from team members

### During Review

1. **Address feedback:** Commit changes normally (don't squash—reviewers see the conversation)
   ```bash
   git add .
   git commit -m "Address review feedback"
   git push
   ```

2. **After approval, rebase if needed:**
   ```bash
   git rebase -i origin/main  # Optional: squash into logical commits
   git push --force-with-lease
   ```

### Merging

- **Maintainers merge with squash merge** to keep main history clean
- Original PR link is preserved for detailed history
- Branch is automatically deleted

### After Merge

```bash
git checkout main
git pull origin main
git branch -d feat/my-feature
git push origin --delete feat/my-feature
```

## 🎯 Code Review Guidelines

### For Authors
- Keep PRs focused (one feature per PR)
- Keep PRs small (<500 lines of changes when possible)
- Request reviews early, don't wait until "done"
- Respond to feedback promptly
- Test locally before pushing

### For Reviewers
- Look for correctness, clarity, and adherence to standards
- Approve quickly if no issues found
- Be specific in feedback ("this variable name is unclear" not "bad naming")
- Approve explicitly before merge

## ✅ Pre-Commit Checklist

- [ ] Branch is rebased on latest `origin/main`
- [ ] Commit messages follow Conventional Commits format
- [ ] Tests pass locally (`pytest tests/`)
- [ ] Code follows project style (use linters if available)
- [ ] No debug print statements or commented code
- [ ] PR description explains *why*, not just what

## 🚨 Important Rules

1. **Never force-push to `main`** or shared branches
2. **Always rebase before PR** to keep history linear
3. **Keep branches short-lived** (target: merge within 2-3 days)
4. **Link to issues** in PR and commit messages
5. **Request reviews explicitly** don't assume team sees your PR

## 🆘 Getting Help

- **Merge conflicts?** Ask in PR comments, we'll help resolve
- **Unsure about branch name?** Check the examples above or ask
- **Git help?** Run `git help <command>` or ask the team

## 📚 Resources

- [Conventional Commits](https://www.conventionalcommits.org/)
- [Trunk-Based Development](https://trunkbaseddevelopment.com/)
- [Git Rebase Tutorial](https://git-scm.com/book/en/v2/Git-Branching-Rebasing)
- [Interactive Rebase Guide](https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History)

Thanks for keeping our repo clean and organized! 🎉
