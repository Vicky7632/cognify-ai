# Instructions for AI coding agents

- Before making changes, read `CONTEXT.md` and the relevant documents in `docs/`; inspect the existing code, repository status, and project conventions.
- Implement only the requested change and the active roadmap phase. Do not make unrelated edits, introduce speculative abstractions, or implement features marked proposed/planned without an explicit task.
- Preserve existing user changes. Do not modify LectureIQ.
- Keep provider-specific code replaceable where practical. Validate uploads, configuration, external responses, and failure cases; surface errors rather than silently returning success-shaped fallbacks.
- Never hardcode or expose API keys. `.env.example` may contain placeholders only; do not commit `.env` or real credentials.
- Follow the documented v1 scope: no authentication (not safe for public exposure), no reranking, and no claims that hallucinations are eliminated.
- Add or update focused tests for behavior and failures. Run the smallest relevant test, lint, or type-check command available; do not claim checks that were not run.
- Update directly related documentation when behavior or decisions change.
- In the final report, list exact files changed and the validation commands/results; distinguish implemented behavior from plans and unresolved choices.
