# Evaluation plan

> **Status:** Evaluation protocol to implement. No evaluation set, benchmark result, or performance claim is provided by this document.

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

## Procedure and acceptance gates

1. Freeze a versioned dataset, split, annotation, and metric definitions before tuning.
2. Record exact software revision, chunking parameters, embedding model/version, vector index settings, retrieval top-k/threshold, prompt version, and LLM/transcription model versions.
3. Run extraction and retrieval tests, then grounded generation and abstention tests. Preserve raw metric inputs and aggregate results without retaining sensitive content unnecessarily.
4. Compare each change against a baseline on the same held-out set. Set numeric release thresholds only after an initial measured baseline and product risk review; no target scores are assumed here.
5. Investigate regressions by modality, source location, and failure category. Do not enable reranking in v1; consider it only if measured retrieval deficiencies justify a separate evaluation.

## Limitations

Offline datasets may not represent real users or adversarial inputs. Similarity scores are not calibrated evidence of truth. LLM judging alone is not a reliable ground truth. Retrieval, citations, and abstention can reduce unsupported responses but cannot eliminate hallucinations. Evaluation does not replace security, privacy, licensing, or operational review.
