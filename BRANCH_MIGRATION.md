# Branch Migration Guide

## Summary
After renaming `master` → `main`, you have two active feature branches to migrate:

### LSTM-core
- **Status**: 5 new commits beyond main (last: `fa478f6 run_setup`)
- **Action**: Rebase on main, then keep or merge based on team decision

### Supervised  
- **Status**: 12 new commits beyond main (last: `e486a21 Merge branch...`)
- **Action**: Rebase on main, then keep or merge based on team decision

### generators & tracker
- **Status**: ✅ Already merged into main (marked as `[gone]`)
- **Action**: Safe to delete after verification

---

## Option A: Rebase & Continue (Recommended)

Use this if you're actively developing on these branches.

### For LSTM-core:
```bash
git fetch origin
git checkout LSTM-core
git rebase origin/main
# Handle any conflicts that arise
git push --force-with-lease
```

### For Supervised:
```bash
git fetch origin
git checkout Supervised
git rebase origin/main
# Handle any conflicts that arise
git push --force-with-lease
```

---

## Option B: Merge & Close (If Branch is Stale)

Use this if the branch is complete and ready to integrate.

### For LSTM-core:
```bash
git checkout main
git pull origin main
git merge --no-ff LSTM-core    # Creates merge commit (preserves history)
git push origin main
git branch -d LSTM-core
git push origin --delete LSTM-core
```

### For Supervised:
```bash
git checkout main
git pull origin main
git merge --no-ff Supervised
git push origin main
git branch -d Supervised
git push origin --delete Supervised
```

---

## Option C: Delete (If Branch is Obsolete)

Use this only if the branch is no longer needed.

```bash
git branch -D LSTM-core              # Delete locally
git push origin --delete LSTM-core   # Delete remote

git branch -D Supervised
git push origin --delete Supervised
```

---

## Recommended Actions

### 1. Clean up merged branches
```bash
git checkout main
git pull origin main

# Delete generators (already merged)
git branch -d generators
git push origin --delete generators

# Delete tracker (already merged)
git branch -d tracker
git push origin --delete tracker
```

### 2. Decide on LSTM-core and Supervised
- **Talk to your team** about the status of these branches:
  - Are they actively developed?
  - Are they ready to merge?
  - Should they be rebased on main?

### 3. Once decided, execute the chosen option above

---

## Verification Checklist

After migration, verify:
```bash
# Check no old master refs exist
git branch -a | grep -E 'master|origin/master'  # Should be empty or only show old refs

# Verify main is default
git symbolic-ref refs/remotes/origin/HEAD

# List current branches
git branch -a
```

Expected output should show:
- ✅ main (your default)
- ✅ LSTM-core (or deleted)
- ✅ Supervised (or deleted)
- ❌ No master or origin/master
