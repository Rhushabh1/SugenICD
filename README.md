# SugenICD
AI-powered Clinical Coding Engine

## Development score and expected performance

Best observed dev score is 40.83/75 measured using the supplied score.py on the 60 dev notes.

My rough estimate for hidden evaluation is 45-50/100. This is only an estimate as the dev set was small and the hidde set includes different clinical distributions and multilingual notes, so i expect generalisation to be harder.

## Data

I used the supplied ICD-11 MMS 2026-01 catalogue and the 60 dev notes. I also added an enriched catalogue mms-2026-01-enriched.json using additional WHO ICD-11 information such as index terms, inclusion terms, definitions, etc. The WHO enrichment was used to improve retrieval while the final output codes are checked against the official MMS catalogue

## Approach and Evaluation

The pipeline combines SQLite FTS5/BM25 lexical retrieval with Fastembed semantic retrieval using "BAAI/bge-small-en-v1.5". Their rankings are combined using weighted RRF (reciprocal rank). The top 50 candidates are prepared and filtered to remove non-leaf categories before being passed to the LLM reranker.

The frozen dev configuration uses title-only semantic embeddings, equal lexica and semantic retrieval weights and Ollama's "qwen3:1.7b" for reranking. The final ranking combines retrieval and LLM rankings using weighted RRF.

I evaluated lexical, semantic and hybrid retrieval separately, compared retrieval hit rates at different cut-offs, and tested LLM reranking and final fusion on the 60 dev notes. I also ran docker-based input/output and local ollama checks.

## AI tooling issue caught

The LLM reranker initially selected a broad parent category (like 1F40) instead of more specific supported code (like 1F40.Z in dev-001 note). I checked the catalogue hierarchy and found that the parent-child mapping was not being handled correctly. I fixed the hierarchy handling and added leaf-node filtering before reranking, rather than relying only on more prompt instructions (which are counter-productive and redundant in my experience).

## Time spent

Total time spent = approx 15 hrs (dedicated)

