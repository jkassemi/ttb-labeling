* Code is **read more than written**; clarity > cleverness. ([Python Enhancement Proposals (PEPs)][1])
* Follow a community standard (PEP 8) as baseline. ([Python Enhancement Proposals (PEPs)][1])
* Consistency within this project > external norms.

## Core Conventions

### Typing

* Use **built-in generics** (`list[int]`, `dict[str, Foo]`).
* Use **PEP 604 unions** (`int | None`).
* All public APIs **must have full type signatures** (args + return).
* Avoid unused `typing` imports.

### Imports & Structure

* Group imports: stdlib → third-party → local.
* One import per line unless logically grouped.

### Naming

* `snake_case` for functions/vars; `CamelCase` for classes.
* Be descriptive; avoid abbreviations.

### Functional Patterns

* Prefer **pure functions** with explicit inputs/outputs.
* Favor **expressions over statements** (comprehensions, generator expressions).
* Use **generators** for large/streamed data.

### Mutability

* Do not mutate inputs in pure functions.
* Favor `tuple`/`frozenset` for stable sequences.

## Toolchain & Checks

### Format & Lint

* Use `ruff` for format + lint + import sorting.
* Always run:

  ```
  ruff . --fix
  ruff .
  ```
* Enforce line length consistent with tooling defaults.

### Type Checking

* `ruff` (or `pyright`) must succeed with no errors.

### Tests

* Use `pytest`.
* Tests reside under `tests/`.
* Quick feedback: focus on unit coverage and edge conditions.

## Docstrings

* Use **Google docstring** style.
* Do not repeat types in docstrings; use them for behavior and examples.

## Prohibitions

* Don’t use `typing.List`/`typing.Dict`.
* Don’t suppress errors without context.
* Don’t introduce dependencies without approval.
