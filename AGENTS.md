---

# AGENTS.md – Operational Rules for Codebase Integrity

These rules define how agents must operate on the codebase to ensure it remains **self-testing**, **well-documented**, and **clean**.

---

## 1. ✍️ Document All Code with Docstrings

.``Whenever you **create or modify** a function or class, you **must** include a concise docstring.

### For Functions

The docstring must follow this structure:

1.  **Purpose**: A brief, one-line summary of what the component does.
2.  **Arguments** (`Args:`): A list of all arguments, their expected types, and a short description.
3.  **Returns** (`Returns:`): A description of the value(s) returned by the function.

### For Classes

When documenting a **class**, you must provide comprehensive documentation that includes:

1.  **Class Docstring**:
    *   A brief, one-line summary of the class's purpose.
    *   `Args:`: A list of all arguments for the `__init__` method, their types, and descriptions.
    *   `Attributes:`: A list of all class attributes (i.e., `self.attribute`), their types, and descriptions.
2.  **Method Docstrings**:
    *   Every method within the class must have its own docstring, following the same structure as for standalone functions.

### Example (Function):

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

### Example (Class):
```python
class DataProcessor:
    """
    Processes raw data by cleaning and transforming it.

    Args:
        raw_data (list[dict]): A list of raw data records.
        transformation (str): The name of the transformation to apply.

    Attributes:
        data (list[dict]): The processed data.
        records_count (int): The number of records.
    """
    def __init__(self, raw_data: list[dict], transformation: str):
        self.records_count = len(raw_data)
        self.data = self._transform(raw_data, transformation)

    def _transform(self, data: list[dict], transformation: str) -> list[dict]:
        """
        Applies a specific transformation to the data.

        Args:
            data (list[dict]): The data to transform.
            transformation (str): The transformation type.

        Returns:
            list[dict]: The transformed data.
        """
        # ... implementation ...
        return data

    def get_record_count(self) -> int:
        """Returns the number of data records."""
        return self.records_count
```

---

## 2. 🧾 Sync Function Documentation with `README.md`

To ensure the high-level documentation is always current, every time you **add or modify a function**, you must also update its entry in `README.md`.

*   **What to include**: The function's signature, a brief description of its purpose, and a simple usage example. The information from the function's docstring is a good source for this.
*   **Goal**: Keep the `README.md` file as a human-readable, quick reference for the project's key functionalities.

---

## 3. 🧪 Self-Test All Changes with `pytest`

Whenever you **introduce or modify** any functionality:

* Whenever a **function** or **class** is created or modified, the agent should **automatically write a corresponding test**. This test should be placed in a **new file inside the `tests/` directory**, and it must **import the relevant function or class**.
* You should then:
    1. **Write example usage** of the function or class.
    2. **Use assertions and print statements** to verify the behavior.
    3. If anything looks suspicious or unexpected, you should **revisit the original code** and investigate. You edit the function or class you created again and go through the same process until it works as expected. You then delete the test file you made and put the test code you just written inside the `tests/functionalities_test.py`, inside a function which will work with pytest.
* You can use any command you want (e.g., `python ...`, `pytest`) to run your tests.
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

## 7. 🔍 Review Existing Code for Context and Conventions

Before making **any modification** to an existing file or introducing new code in an established area:

*   **Read surrounding code**: Understand the local context, including imports, function/class definitions, and variable usage.
*   **Identify conventions**: Observe existing patterns for naming, formatting, error handling, and architectural choices.
*   **Mimic style**: Ensure your changes integrate naturally and idiomatically with the surrounding code.
*   **Goal**: Maintain consistency across the codebase, making it easier for others to understand and contribute, and preventing the introduction of divergent styles or anti-patterns.

---

## ✅ Example Workflow

> 🔧 You introduce a new function `def normalize(vec): ...`

* Add a docstring explaining its purpose, arguments, and return value.
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
* **Well-documented** through docstrings and `README.md`
* **Minimal and truthful** through careful removals
* **Reliable** by ensuring every commit leaves the test suite in a passing state
* **Context-sensitive** in all task planning and recommendations

---