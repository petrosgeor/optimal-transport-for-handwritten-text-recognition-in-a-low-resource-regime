---

# AGENTS.md – Operational Rules for Codebase Integrity

These rules define how agents must operate on the codebase to ensure it remains **self-testing**, **well-documented**, and **clean**.

---

## Planning

Before responding, please:

1. Decompose the request into core components
2. Identify any ambiguities that need clarification
3. Create a structured approach to address each component
4. Validate your understanding before proceeding

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
3.  **README.md Documentation for Classes**:
    *   Every class documented in `README.md` must have an **Attributes** and a **Methods** section.
    *   In the **Methods** section, you must enumerate the methods of the class in bullet points.
    *   **Important**: To avoid redundancy, class methods must *only* be documented under their class's **Methods** section. They must **not** be documented again as separate functions elsewhere in `README.md`.



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

## 3. 🧪 Self-Test All Changes via Command Line

Whenever you **introduce or modify** any functionality, validate it using
ephemeral toy examples executed in the command line. Do not create, modify,
or delete any files for these checks.

- Write a minimal example that imports the updated function/class and exercises
  its core behavior. Prefer a single command such as:
  - `python - <<'PY' ... PY`
  - or `python -c "..."`
- Use simple `print` statements and/or inline `assert` statements to confirm
  expected outcomes. The example must not write to the repository or leave
  artifacts on disk.
- If outputs are unexpected, iterate on the implementation and re-run the same
  command-line example until it behaves as intended.
- Do not add or edit files under `tests/` and do not use `pytest`.

---

## 4. ▶️ Run Checks and Report Results

After making a change and preparing a toy example:

- **Run the command-line example** that validates the change.
- **Report the result** by pasting the exact command and its output, stating:
  - What was expected, and
  - What was observed (including any traceback if it failed).

Do **not** consider the task complete until the command-line example clearly
demonstrates correct behavior.

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
* Validate via a toy command-line example (no file writes):

```bash
python - <<'PY'
from your_module import normalize
v = normalize([3, 4])
print('result:', v)
assert all(abs(a-b)<1e-6 for a,b in zip(v, [0.6, 0.8]))
print('OK')
PY
```

* If the assertion fails, refine the implementation and re-run the same
  example until it prints the expected result and `OK`.

---

## 🎯 Goal

Keep the repo:

* **Self-verifying** through lightweight command-line checks
* **Well-documented** through docstrings and `README.md`
* **Minimal and truthful** through careful removals
* **Reliable** by ensuring every commit leaves the test suite in a passing state
* **Context-sensitive** in all task planning and recommendations

---
