# SugenICD
AI-powered Clinical Coding Engine

## Development score and expected performance

Best observed dev score is 40.83/75 measured using the supplied score.py on the 60 dev notes.

My rough estimate for hidden evaluation is 45-50/100. This is only an estimate/speculation - the dev set was small and the hidden set includes different clinical distributions and multilingual notes, so i expect generalisation to be harder.

## Data

I used the supplied ICD-11 MMS 2026-01 catalogue and the 60 dev notes. I also added an enriched catalogue mms-2026-01-enriched.json using additional WHO ICD-11 information such as index terms, inclusion terms, definitions, etc. The WHO enrichment was used to improve retrieval while the final output codes are checked against the official MMS catalogue

For multilingual expansion, I added -> 
1) bharatavani/CSTT hindi clinical glossary
2) compact marathi-english terminology dictionary extracted from a publicly available marathi-english dictionary
3) manually curated hindi, marathi, and hinglish clinical phrase mappings

The dictionaries provide English clinical terminology hints -> they do not replace or translate the original note (baseline pipeline is undisturbed)

## Approach and Evaluation

The pipeline combines SQLite FTS5/BM25 lexical retrieval with Fastembed semantic retrieval using "BAAI/bge-small-en-v1.5". Their rankings are combined using weighted RRF (reciprocal rank). The top 50 candidates are prepared and filtered to remove non-leaf categories before being passed to the LLM reranker.

The frozen dev configuration uses title-only semantic embeddings, equal lexica and semantic retrieval weights and Ollama's "qwen3:1.7b" for reranking. The final ranking combines retrieval and LLM rankings using weighted RRF.

I evaluated lexical, semantic and hybrid retrieval separately, compared retrieval hit rates at different cut-offs, and tested LLM reranking and final fusion on the 60 dev notes. I also ran docker-based input/output and local ollama checks.

Beyond the official dev set, I created a custom 18-note multilingual test set containing poor english clinical descriptions and hindi, marathi, hinglish and transliterated terminology. The Ollama pipeline achieved 38.89% Top-1 and 55.56% Top-3, compared with 27% Top-1 and 50% Top-3 for offline retrieval.

This was a small, synthetic test set used to check multilingual terminology expansion and compare pipeline behaviour and it is not an independently clinically validated benchmark.

## AI tooling issue caught

The LLM reranker initially selected a broad parent category (like 1F40) instead of more specific supported code (like 1F40.Z in dev-001 note). I checked the catalogue hierarchy and found that the parent-child mapping was not being handled correctly. I fixed the hierarchy handling and added leaf-node filtering before reranking, rather than relying only on more prompt instructions (which are counter-productive and redundant in my experience).

## Time spent

Total time spent = approx 16-17 hrs (dedicated)

