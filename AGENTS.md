// put in front of you a man who knows nothing. others say I do. but they're expert in their own ways. i'm past the point of universal understanding of what i'm doing. but i'm terrible at communication. i hope my children are better than i am. please help me stay on track and complete this quickly and well.

* Code is **read more than written**; clarity > cleverness.
* Use **PEP 8** as the baseline.
* Consistency within this project > external norms.

## Core Conventions

### Language

* Prefer features available in our supported Python versions.
* Supported Python: 3.12+.
* Use `from __future__ import annotations` only when needed for our version support / forward refs.

### Typing

* Use **built-in generics** (`list[int]`, `dict[str, Foo]`).
* Use **PEP 604 unions** (`int | None`).
* All public APIs **must have full type signatures** (args + return).
* Prefer `collections.abc` for inputs (`Sequence`, `Mapping`, `Iterable`) and concrete return types (`list`, `dict`).
* Use `Protocol` to make external systems injectable (OCR engines, HTTP clients, DB adapters).
* Mark module constants as `Final` when helpful.
* Avoid unused `typing` imports.
* **Ruff is not a type checker**; use `pyright` (or `mypy`) for type checking. ([Astral Docs][1])

### Imports & Structure

* Group imports: stdlib → third-party → local.
* One import per line unless logically grouped.
* Avoid import-time side effects (env vars, network calls, GPU init). Keep modules importable in tests.
* Local imports are OK for optional/heavy deps; add a short comment (“optional dependency”, “startup perf”).

### Naming

* `snake_case` for functions/vars; `CamelCase` for classes; `UPPER_SNAKE_CASE` for constants.
* Be descriptive; avoid abbreviations unless widely standard in-domain.

### Functions & Boundaries

* Separate **IO/backends** (OCR, filesystem, network) from **pure parsing/logic**.
* Prefer straightforward control flow; use comprehensions only when they’re simpler than a loop.
* Use generators/iterators for large/streamed data when it improves memory use *without* harming readability.

### State & Caching

* Avoid module-level **mutable** state (singletons, caches).
* For expensive factories, prefer `@lru_cache(maxsize=1)` or explicit dependency injection.
* Ensure cached backends are safe for concurrency, and provide test reset seams when needed.

### Mutability

* Do not mutate inputs.
* Use immutable containers for constants/config (`tuple`, `frozenset`).
* Prefer small normalization helpers + mapping tables over nested conditionals.

### Errors & Logging

* Don’t suppress errors without context.
* Bubble up parser/IO errors to the system boundary so they surface in the structured error response.
* Catch broad exceptions only at **system boundaries**; inside core logic, raise domain-specific exceptions.
* When catching at a boundary: log with context (`exc_info=True`), return stable error codes, avoid leaking sensitive data.
* If re-raising, use `raise ... from exc` to preserve cause.
* Avoid "fallback on failure" code paths; this includes switching to a second extraction path when the primary path fails. If multiple extraction paths are used, run them together and cross-verify outputs (e.g., document understanding proposes a statement and OCR verifies it), and make that behavior explicit and testable via configuration.

## Toolchain & Checks

### Format & Lint (Ruff)

* Local autofix + format: run `ruff check --fix` then `ruff format`. ([Astral Docs][2])
* Import sorting requires lint fix (formatter does not sort imports): run `ruff check --select I --fix` then `ruff format`. ([Astral Docs][3])
* CI check mode: run `ruff check` and `ruff format --check`. ([Astral Docs][2])
* In pre-commit/CI ordering, apply lint fixes before formatting. ([Astral Docs][4])
* Pin Ruff behavior with `required-version` in config to avoid drift across environments. ([Astral Docs][5])

### Type Checking

* `pyright` (or `mypy`) must succeed with no errors.

### Tests

* Use `pytest`.
* Prefer fast unit tests; inject/mock external backends (OCR, network, filesystem).
* Tests reside under `tests/`.

## Docstrings

* Use **Google docstring** style.
* Do not repeat types in docstrings; document behavior, invariants, side effects, and edge cases.

## Prohibitions

* Don’t use `typing.List` / `typing.Dict`.
* Don’t add blanket `# noqa` / `type: ignore` without a reason (and scope it narrowly).
* Don’t introduce dependencies without approval.
* Don’t introduce global mutable singletons/caches for external backends.
* Don’t mutate environment variables inside business logic.

[1]: https://docs.astral.sh/ruff/faq/?utm_source=chatgpt.com "FAQ | Ruff - Astral Docs"
[2]: https://docs.astral.sh/ruff/linter/?utm_source=chatgpt.com "The Ruff Linter - Astral Docs"
[3]: https://docs.astral.sh/ruff/formatter/?utm_source=chatgpt.com "The Ruff Formatter - Astral Docs"
[4]: https://docs.astral.sh/ruff/integrations/?utm_source=chatgpt.com "Integrations | Ruff - Astral Docs"
[5]: https://docs.astral.sh/ruff/settings/?utm_source=chatgpt.com "Settings | Ruff - Astral Docs"
