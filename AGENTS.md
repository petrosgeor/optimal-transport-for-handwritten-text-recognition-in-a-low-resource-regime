# AGENTS.md – Operational Rules for Codebase Integrity

These rules define how agents must operate on the codebase to ensure it remains **self-testing**, **well-documented**, and **clean**.

---

## 1. 🧾 Sync the Docs

Whenever you **add**, **remove**, **rename**, or **modify** any class, function, or script:

* **Update the corresponding section in `README.md` in the same commit.**
* This includes docstring or signature changes, not just new files.

---

## 2. 🧪 Self-Test All Changes with `pytest`

Whenever you **introduce or modify** any functionality:

* Create or update a **minimal working test** that exercises its **core behavior**.
* Use the `pytest` framework exclusively.
* Always add or update tests in the file:
  `tests/functionalities_test.py`.

### If `tests/functionalities_test.py`:

* **Does not exist** → Create it and include at least one test for the new/changed functionality.
* **Exists** → Append relevant new test(s) in the same file.

### If a test becomes **obsolete** (e.g., due to interface change or removed behavior):

* **Remove it**, as long as it no longer represents the updated truth and would raise an invalid assertion.

### Test expectations:

* **Use `assert` statements** to check meaningful output.
* Test **core logic only** (no edge-case overfitting).
* **Commit the test** in the **same commit** as the corresponding code change.

---

## 3. ▶️ Run the Tests and Report Results

After making any change and writing/updating the tests:

* **Run all tests** using `pytest`.
* **Report the result**:

  * Confirm **all tests passed**, or
  * Identify and explain any test failures (whether from new or existing tests).

Do **not** consider the task complete until all tests pass and the test suite is clean.

---

## 4. 🔁 Avoid Redundant Work

Before implementing any change:

* Check whether the requested functionality **already exists** in the codebase.
* If it does, do **not** re-implement it.
* Instead, **explain** that the functionality is already implemented and reference where it exists.

---

## ✅ Example Workflow

> 🔧 You introduce a new function `def normalize(vec): ...`

* Update `README.md` to reflect the new function.
* Add a test like:

```python
def test_normalize():
    assert normalize([3, 4]) == [0.6, 0.8]
```

* Save this inside `tests/functionalities_test.py`. Create the file if needed.
* Remove any obsolete tests if needed.
* Run `pytest` and confirm all tests pass.
* Report the test result clearly.

---

## 🎯 Goal

Keep the repo:

* **Self-verifying** through tests
* **Up-to-date** through docs
* **Minimal and truthful** through careful removals
* **Reliable** by ensuring every commit leaves the test suite in a passing state

