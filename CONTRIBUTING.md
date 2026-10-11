# Git & Development Workflow Guidelines

This document defines the version control policy and commitment rules for the **Vachan** project.

---

## 📌 Strict Rule: Commit Early, Commit Often

**Every change—regardless of how small—must be committed and pushed to GitHub.**

### Guidelines:
1. **No Uncommitted Work Left Behind**:
   - Every file modification, refactor, bug fix, dependency update, documentation improvement, or new feature step must have its own Git commit.
   - Do not batch large numbers of unrelated small changes together into a single delayed commit.

2. **Atomic & Frequent Commits**:
   - Completed a single component or function? **Commit it.**
   - Fixed a typo or updated docstrings? **Commit it.**
   - Adjusted a setting in `settings.json`? **Commit it.**
   - Added a unit test? **Commit it.**

3. **Clear Commit Messages**:
   - Use concise and descriptive conventional commit prefixes:
     - `feat:` for new capabilities or pipeline steps
     - `fix:` for bug fixes
     - `refactor:` for code restructuring without changing behavior
     - `docs:` for documentation updates
     - `style:` for formatting and lint fixes
     - `chore:` for config, dependencies, and environment scripts
     - `test:` for writing or updating test suites

4. **Always Sync with GitHub**:
   - Regularly push commits to the remote GitHub repository so that project progress is backed up and transparently tracked.
   - Keep the working directory clean (`git status` should show no untracked or uncommitted working files at the end of every task).

---

> **Note for AI Coding Assistants:**
> Whenever making edits to this codebase, proactively create a commit for every discrete step or task, ensuring full Git traceability for all modifications.
