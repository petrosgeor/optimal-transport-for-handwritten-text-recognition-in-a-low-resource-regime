---

# AGENTS.md – Operational Rules for Codebase Integrity

These rules define how agents must operate on the codebase to ensure it remains **self-testing**, **well-documented**, and **clean**.

---

## 1. 🧠 Understand the Codebase via the Knowledge Graph

Before performing **any** task, you **must** first read and understand the project's structure by consulting the knowledge graph:

*   **File Location**: `overview/knowledge_graph.json`

This file is the single source of truth for the codebase architecture. It contains all the information about modules, classes, functions, and their relationships. Use it to gather context and understand how the project works before making any changes.

---

## 2. ✍️ Document All Code with Docstrings

Whenever you **create or modify** a function or class, you **must** include a concise docstring that follows this structure:

1.  **Purpose**: A brief, one-line summary of what the component does.
2.  **Arguments** (`Args:`): A list of all arguments, their expected types, and a short description.
3.  **Returns** (`Returns:`): A description of the value(s) returned by the function.

This ensures that the `overview/knowledge_graph.json` remains populated with useful, human-readable context.

### Example:

```python
def normalize_vector(vector: list[float]) -> list[float]:
    """
    Normalizes a vector to have a unit length of 1.

    Args:
        vector (list[float]): The input vector to normalize.

    Returns:
        list[float]: The normalized vector.
    """
    # ... implementation ...
```

---

## 3. 🧪 Self-Test All Changes with `pytest`

Whenever you **introduce or modify** any functionality:

* Whenever a **function** or **class** is created or modified, the agent should **automatically write a corresponding test**. This test should be placed in a **new file inside the `tests/` directory**, and it must **import the relevant function or class**.
* The agent should then:
    1. **Write example usage** of the function or class.
    2. **Use assertions and print statements** to verify the behavior.
    3. If anything looks suspicious or unexpected, the agent should **revisit the original code** and investigate. He edits the function or class he created again and goes through the same process until it works as expected. He then deletes the test file he made and puts the test code he just written inside the `tests/functionalities_test.py`, inside a function which will work with pytest.
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

## 4. ▶️ Run the Tests and Report Results

After making any change and writing/updating the tests:

* **Run all tests** using `pytest`.
* **Report the result**:

  * Confirm **all tests passed**, or
  * Identify and explain any test failures (whether from new or existing tests).

Do **not** consider the task complete until all tests pass and the test suite is clean.

---

## 5. 🔁 Avoid Redundant Work

Before implementing any change:

* Check whether the requested functionality **already exists** in the codebase.
* If it does, do **not** re-implement it.
* Instead, **explain** that the functionality is already implemented and reference where it exists.

---

## 6. 📋 Propose Specific Tasks with Context

When proposing a new task:

* Be **very specific** about what needs to be done.
* If the user has provided **code snippets**, **pseudo code**, or **partial implementations** in the conversation:

  * **Include those in the proposed task description.**
  * Clearly indicate **how** the user-provided code is to be integrated, adapted, or extended.
* Avoid generic task descriptions; always ground your suggestions in the actual context of the conversation and the codebase.

This ensures that proposed tasks are **directly actionable**, **context-aware**, and aligned with the user's intentions and inputs.

---

## ✅ Example Workflow

> 🔧 You introduce a new function `def normalize(vec): ...`

* Add a docstring explaining its purpose, arguments, and return value.
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
* **Well-documented** through docstrings
* **Minimal and truthful** through careful removals
* **Reliable** by ensuring every commit leaves the test suite in a passing state
* **Context-sensitive** in all task planning and recommendations

---
