# Contributing

Thanks for helping build Cognify AI. The current repository is a documentation baseline; implementation should follow the phases in [ROADMAP.md](ROADMAP.md).

## Before changing files

1. Read [CONTEXT.md](CONTEXT.md), this contribution guide, and relevant documents under [`docs/`](docs/).
2. Inspect the current repository and status; preserve unrelated and user-authored changes.
3. Confirm the active roadmap phase and keep a change scoped to that phase.
4. Do not modify LectureIQ, add real credentials, or describe planned behavior as implemented.

## Proposed development expectations

The project intends to use Python, FastAPI, Pydantic, and pytest, but supported Python version, package manager, and exact lint/type-check tools have not yet been selected. Once established, use the repository's documented setup and targeted checks rather than introducing another tool without need.

For implementation contributions:

- Include focused tests for normal behavior, validation, and relevant dependency failures.
- Validate uploaded media and configuration; surface errors explicitly.
- Keep embedding, transcription, vector-store, and LLM provider details replaceable where practical.
- Preserve source provenance through extraction, chunking, retrieval, and citations.
- Update directly related documentation and identify new decisions.
- Do not add reranking to v1 without a separately evaluated decision.

## Pull requests

Use the [pull request template](.github/pull_request_template.md). Describe behavior, tests actually run, configuration or migration impacts, security/privacy considerations, and any unresolved choice. Do not claim performance, quality, or implemented features without evidence.
