# Frontend source layout

No TypeScript application modules are implemented yet. Add `App.tsx` and `main.tsx` only when the frontend implementation phase begins; do not create empty source files.

Planned responsibilities:

- `assets/`: imported static assets.
- `components/`: reusable UI components.
- `features/`: user-facing feature areas such as document management and chat.
- `hooks/`: shared React hooks.
- `layouts/`: page-level layout composition.
- `lib/`: framework/client setup and shared low-level integrations.
- `pages/`: route-level screens.
- `services/`: Cognify `/api/v1` HTTP client calls; never store provider secrets here.
- `types/`: shared TypeScript contracts.
- `utils/`: small, general-purpose client utilities.

Future UI flows include document upload and processing status, a document library/source management, AI chat with citations, short/detailed summaries, topic-wise notes/key concepts, and timestamped audio/video transcripts.
