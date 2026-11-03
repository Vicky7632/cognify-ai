# Evaluation plan

> **Status:** Evaluation protocol documented; offline Recall@k and MRR@k calculation helpers and a small versioned synthetic dataset are implemented. No benchmark result or performance claim is provided by this document.

## Goals

Measure whether retrieval finds relevant evidence, whether answers and citations are supported by that evidence, whether the system abstains for unanswerable questions, and whether failures are surfaced correctly. Track model, prompt, index, chunking, and dataset versions so comparisons are reproducible.

## Dataset design

Create a small, versioned development set and a separately held-out test set from documents with permission to use. Include PDF pages, TXT, audio transcripts, and video transcripts when the corresponding modality is implemented. Record for each query:

- expected relevant source/chunk identifiers and page/time locations;
- whether evidence is sufficient to answer;
- a reference answer or key facts for answerable queries;
- query type and modality/source type.

Include paraphrases, multi-chunk questions, metadata-filtered retrieval, ambiguous questions, and questions whose answer is absent. Avoid leaking test answers into prompts or tuning.

## Measurable metrics

Report counts and denominators as well as rates. Compute metrics per modality and in aggregate; do not aggregate away a weak modality.

- **Retrieval Recall@k:** queries with at least one labeled relevant chunk in top-k / answerable retrieval queries.
- **MRR@k:** mean reciprocal rank of the first relevant chunk, with zero for no relevant result.
- **Citation location accuracy:** returned citations whose source and page/time overlap a labeled supporting location / all evaluated returned citations.
- **Citation coverage:** supported answer claims with at least one valid citation / all answer claims requiring factual support, using a documented human or annotation rubric.
- **Answer support rate:** answers judged fully supported by retrieved evidence / all non-abstained answers, assessed against a fixed rubric by reviewers.
- **Abstention precision and recall:** on the labeled unanswerable/sufficient-evidence split, report correct abstentions and incorrect abstentions separately; define the positive class explicitly in the implementation report.
- **Ingestion correctness:** successful test fixtures with expected extracted text and provenance preserved / fixtures expected to process successfully; report errors separately by type and stage.
- **Failure behavior:** tests for malformed/unsupported/oversized files and unavailable extractors, embedding, vector store, and LLM must produce the specified explicit error behavior.

Use confidence intervals or report sample counts for small sets; do not present unstable rates as general guarantees. Human judgment needs a written rubric and, for consequential comparisons, independent review or adjudication.

## Latency and failure cases

Measure latency only on a documented environment and representative dataset. Separate upload, extraction/transcription, chunking/embedding, vector search, and generation timings. Record sample counts, percentile methodology, hardware, model/runtime versions, input sizes, concurrency, and whether cold-start/model-load time is included. No target or result is asserted here.

Exercise malformed/unsupported/oversized files, empty extraction/transcripts, parser/FFmpeg/model failures, PostgreSQL outages, file-storage failures, Pinecone errors/timeouts, embedding/LLM errors, and invalid citations. Verify failures are explicit, processing state remains coherent, sensitive content and secrets are not leaked, and no failed request returns a success-shaped result.

## Procedure and acceptance gates

1. Freeze a versioned dataset, train/dev/test split, annotation, and metric definitions before tuning.
2. Record exact software revision, chunking parameters, embedding model/version, Pinecone index settings, top-k/threshold, prompts, and LLM/transcription model versions.
3. Run component tests for extraction, storage, indexing, and retrieval, followed by grounded-answer, citation, and abstention evaluation. Retain only necessary inputs/results and avoid unnecessary sensitive source retention.
4. Compare changes against a baseline on the same held-out set. Set numeric thresholds only after measuring a baseline and reviewing product risk; no target scores or results are asserted here.
5. Keep regression fixtures for known parser/media/provider failures and API behavior. Investigate regressions per modality, source location, and failure category. Do not enable reranking in v1.

## Limitations

Offline datasets may not represent real users or adversarial inputs. Similarity scores are not calibrated evidence of truth. LLM judging alone is not a reliable ground truth. Retrieval, citations, and abstention can reduce unsupported responses but cannot eliminate hallucinations. Evaluation does not replace security, privacy, licensing, or operational review.

## Dataset versioning checklist

Before publishing an evaluation run, record the dataset version, split, annotation revision, and a stable query ID for every example. Keep held-out test examples separate from development data, and record the code revision and metric configuration with each run. Never report benchmark scores until a dataset has been created and the evaluation has actually been executed.

The initial synthetic offline dataset is `backend/data/retrieval-evaluation/v1.json` (`dataset_version` `1.0.0`, `schema_version` `1.0`). It contains a small chunk corpus and answerable/unanswerable queries, including multi-chunk evidence. An answerable example must identify at least one relevant chunk; an unanswerable example must have none. Bump the dataset version when examples or annotations change; increment the schema version when the JSON structure or validation rules change. Validate the bundled dataset with `.\.venv\Scripts\python.exe -m pytest backend\tests\test_retrieval_evaluation_dataset.py` from the repository root. `run_retrieval_dataset_evaluation` in `app.services.retrieval_dataset_evaluator` loads the versioned dataset (or a supplied dataset path) and scores caller-supplied ranked chunk IDs by query ID using the existing Recall@k and MRR@k metrics; it does not execute retrieval. Missing rankings and empty result lists count as misses; unanswerable examples are excluded. Unknown query/chunk IDs, malformed rankings, and datasets with no answerable queries are rejected with `ValueError`. It is offline and does not call providers. For a smaller step, `evaluate_retrieval_dataset` accepts an already-loaded dataset.

`RetrievalDatasetEvaluation.to_dict()` returns deterministic JSON-compatible reporting data: dataset and schema versions, `k`, total/answerable/unanswerable query counts, Recall@k, and MRR@k. The unanswerable count is total queries minus answerable queries. For example, with the bundled dataset and rankings that retrieve relevant evidence for each answerable query at rank 1:

```json
{
  "dataset_version": "1.0.0",
  "schema_version": "1.0",
  "k": 1,
  "total_query_count": 4,
  "answerable_query_count": 3,
  "unanswerable_query_count": 1,
  "recall_at_k": 1.0,
  "mean_reciprocal_rank_at_k": 1.0
}
```

`evaluate_citations` in `app.services.citation_evaluation` consumes the dataset, per-query available retrieved chunk IDs, and existing `GroundedAnswer` objects (reading citation IDs from their `segment_id` fields). Per-query citation validity is `unique cited IDs present in that query's available retrieved IDs / unique cited IDs`; aggregate validity is the micro-average `total valid unique cited IDs / total unique cited IDs`. Duplicate citations count once. With no cited IDs, validity is `null` because its denominator is zero. Citation coverage is `answerable queries with at least one valid citation / all answerable queries`; unanswerable queries are excluded. If there are no answerable queries, coverage is `null`. Counts are returned with both metrics. ID validity means only that an ID was retrieved; it does not establish that the source supports an answer claim or assess citation location/entailment. Those require separate labeled review.

`evaluate_claim_support` in `app.services.claim_support_evaluation` accepts explicit claim records with a claim ID, text, expected supporting chunk IDs, and actual cited chunk IDs. It marks a claim supported only when every unique expected ID is present among its unique cited IDs. Claim support rate is `supported claims / all claims`; an empty claim set returns `null`, with counts of zero. Duplicate IDs are deduplicated and missing citations leave expected IDs unsupported. This is deterministic ID-set matching against supplied annotations only: it does not verify textual entailment, factual correctness, citation relevance, or semantic support.

The unified offline evaluator can optionally cross-check claim source annotations against generated answer citations using `OfflineRAGEvaluationInput.chunk_id_to_segment_id`, a mapping from dataset chunk labels (for example, `"chunk-break-timing"`) to the corresponding persisted segment UUID. Mapping keys must be known dataset chunk IDs, UUID targets must be unique, and every `actual_cited_chunk_id` used in the comparison must have a mapping. The optional `citation_source_consistency` result compares the deduplicated union of mapped claim source IDs with the answer's deduplicated citation segment IDs for each query with claim annotations. It reports annotated sources missing from the answer, answer citations absent from annotations, and citation UUIDs not represented by the supplied mapping. Repeated citation UUIDs count once. This is a deterministic source-ID consistency check only: `GroundedAnswer` does not associate citations with individual claims, and matching IDs do not prove relevance, entailment, factual correctness, or semantic citation support. When no mapping is supplied, this optional check is omitted; existing citation validity, coverage, and claim-support metrics are unchanged. This mapping is evaluation-only and does not change production API contracts.

`evaluate_abstentions` in `app.services.abstention_evaluation` accepts predictions keyed by the existing dataset query IDs, with a decision of `answered` or `abstained`. Missing predictions are reported separately and never counted as abstentions or correct abstentions. Abstention precision is `correct abstentions / predicted abstentions`; abstention recall is `correct abstentions / all unanswerable queries`; answer coverage is `answered answerable queries / all answerable queries`; incorrect answer rate is `answered unanswerable queries / all unanswerable queries`. Missing predictions remain in the answerable/unanswerable denominators, so they lower coverage or recall without being silently classified as a system decision. Each metric returns `null` when its denominator is zero. This evaluator measures only agreement between the decision and the dataset's sufficient-evidence label; it does not assess answer correctness or semantic support.

`evaluate_offline_rag` in `app.services.offline_rag_evaluation` loads the versioned dataset and orchestrates the retrieval, citation, claim-support, and abstention evaluators above. Provide query-scoped records using existing dataset query IDs, ranked chunk IDs, available retrieved IDs, existing `GroundedAnswer` values, `AnswerClaim` annotations, and `answered`/`abstained` decisions. The wrapper returns the dataset version, each existing evaluator result (including its counts and denominators), per-query citation/abstention results, claim-support results by query, and `missing_inputs`. `OfflineRAGEvaluationResult.to_dict()` produces JSON-compatible output; query results follow dataset order. Run the integration tests from the repository root with `.\backend\.venv\Scripts\python.exe -m pytest backend\tests\test_offline_rag_evaluation.py`.

Missing component inputs are listed by query ID and retain each evaluator's established behavior: missing retrieval rankings are scored as misses; absent answers/citations provide no citation coverage, and absent available-ID lists make cited IDs unavailable for validity checks; unprovided claims are excluded from claim-support scoring; missing decisions remain explicit `missing_prediction` outcomes and are not treated as abstentions. The missing-input report is therefore required context when interpreting partial-run metrics. Empty denominators retain each evaluator's existing `null` behavior. The unified pipeline computes no new metric formulas and makes no provider/network calls. Citation ID validity, claim-to-expected-source matching, and answerability decisions are not evidence of semantic entailment, factual correctness, or actual claim support.
