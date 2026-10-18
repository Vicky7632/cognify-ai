# Cognify AI agent instructions

- Read `CONTEXT.md` and the relevant documentation under `docs/` before coding.
- Inspect existing code, repository status, and conventions before modifying files.
- Make small, focused changes for the active roadmap phase. Preserve unrelated user changes; avoid unrelated edits and unnecessary dependencies/abstractions.
- Do not access or modify LectureIQ. Future integration goes through its existing Express backend.
- Never expose, hardcode, or commit secrets. Keep credentials out of frontend code; `.env.example` has placeholders only.
- Keep PostgreSQL for relational application data, Pinecone for vectors/search metadata, and file storage separate.
- Respect scope: no MVP authentication (never publicly expose without suitable controls); OCR, video-frame analysis, and reranking are not v1 requirements.
- When implementing features, write and run appropriate tests, including failure cases. Never claim unrun checks passed; report exact commands/results.
- Update relevant documentation when behavior or API contracts change. Distinguish implementation from proposals.
- In the final report, list exact files changed, checks performed, and open decisions.
