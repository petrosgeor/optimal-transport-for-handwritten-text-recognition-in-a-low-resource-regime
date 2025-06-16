
# AGENTS.md – Operational Rules for Codebase Integrity & Context-Aware Behaviour  

These rules tell agents **exactly how to behave inside this repository** so it stays:

- **Self-testing** – every change is covered by `pytest`.
- **Well-documented** – `README.md` always matches the code.
- **Clean & minimal** – no dead code or duplicate effort.
- **Context-sensitive** – agents either **plan** or **code**, but **never both at once**.

---

## 🚦 Interaction Contract (Read First!)

1. **Detect intent by keywords**  
   If the user’s request contains **“plan”**, **“task”**, **“proposal”**, or **“road-map”** → You are in **Planning Mode**.  
   Otherwise, assume **Coding Mode** unless the user explicitly says otherwise.

2. **Planning Mode**  
   - Output only a structured **Task Proposal** (see template).  
   - Do **not** write code, commit, or open PRs.  
   - Await approval / changes.

3. **Coding Mode**  
   - Implement the requested code change following the rules below (docs, tests, etc.).  
   - Planning is unnecessary unless the user asks.

4. **If in doubt** – ask the user: **“Do you want a plan or code?”**

---

### 📄 Task Proposal Template (Planning Mode)

task: <Concise title, imperative voice>
context:
  - file: path/to/file.py            # existing files to touch
  - snippet: |                       # user-supplied code, if any
      def example(): ...
steps:
  - Step-by-step bullet list.
acceptance_criteria:
  - Observable, testable checklist.
notes:
  - Optional clarifications or open questions.

---

## 🧭 Implementation Workflow (Coding Mode)

Follow **all** steps for every commit that changes code:

1. **Sync the docs**
   Update `README.md` for any added, removed, renamed, or modified class/function/script.
   This includes changes to docstrings or signatures.
   *(Where: Same commit as the code change)*

2. **Add or update tests**
   Modify `tests/functionalities_test.py` (create the file if it doesn't exist).
   Include `assert`-based checks for core behavior.
   Remove tests **only** if the related behavior was intentionally removed.
   *(Where: Same commit as the code change)*

3. **Run the test suite**
   Execute `pytest` locally and **only commit if all tests pass**.
   (CI should block merges on failure.)

4. **Report test results**
   Include a pass/fail summary in the PR description.

5. **Avoid duplication**
   Search the repo before adding new features.
   If the functionality already exists, link to it and explain why no new code is needed.

If any step fails, fix it before marking the task complete.

---

## ✅ Example (Abridged)

> **User**: “Add a `normalize(vec)` utility.”

**Agent (Coding Mode)**

1. Add `normalize` in `utils/vector.py`.
2. Update `README.md` “Vector utilities” section.
3. Append the following to `tests/functionalities_test.py`:

```python
def test_normalize():
    assert normalize([3, 4]) == [0.6, 0.8]
```

4. Run `pytest` → **all green**.
5. Commit code + docs + test in one commit, push PR, include pass report.

---

## 🔁 Redundancy-Check Details

* If a requested function already exists, **do not** re-implement.
* Respond in *either* Planning Mode (link & propose usage example) *or* Coding Mode (point to existing code, no change needed).

---

## 🎯 Golden Rules

* **One user intent → One agent mode** (Plan *or* Code, never both).
* Every code commit leaves the suite **green** and the docs **accurate**.
* Task proposals are **actionable**, **specific**, and require user approval before code begins.

---


