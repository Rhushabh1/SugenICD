# LOG

Add an entry whenever you make a real choice: an approach, a model, a data source, a rule, a trade-off against the time or money cap, or something you tried and dropped. Each entry gives:

- the date and time, and the commit it goes with
- the decision, and the options you considered
- what you measured or saw that decided it (a score, a failure case, a cost)
- what you would revisit with more time

[10/11/2026, 20:23]
decision -> data inspection + score.py + use supplied data & mms catalogue
alternative -> start preprocessing/retrieval without a validation script or knowledge of data
revisit -> check catalogue hierarchy and child-parent relationships when implementing lexical

commit 1

[10/11/2026, 21:49]
decision -> docker setup with i/o to avoid issues later, keeping it light in dependencies
alternative -> use larger ML image or install all planned AI dependencies upfront
revisit -> add dependencies only when implementing retrieval/models

[10/11/2026, 22:07]
decision -> using python 3.11-slim inside docker while retaining 3.13 for local dev
alternative -> same python version locally and inside docker
revisit -> if a required dependency needs a diff python version

[10/11/2026, 22:34]
decision -> apply norm, lowercase & stripping, whitespace cleanups
alternative -> remove stop words, punctuations, other tokens
revisit -> evaluate more normalizations against retrieval accuracy later

commit 2

[10/11/2026, 23:38]
decision -> use sqlite fts5 with bm25 for lexical retrieval of ICD category candidates from clinical note text. prioritize exact category-title match and derive parent relationships from catalogue uri. also added hit@k metric to establish a baseline first
alternative -> keyword matching only or introduce elasticsearch or another external search tool -> extra lib dependency
revisit -> analyse missed gold codes and improve retrieval before adding semantic retrieval

[11/11/2026, 00:53]
decision -> no truncation of tokens to top 50 & manually expanding clinical & anatomical synonyms for better hit
measurement -> Hit@20: 25/60 = 41.67% improved to now Hit@20: 31/60 = 51.67%
alternative -> use only original note tokens, introduces larger external medical terminology database
revisit -> expand dictionary based on observed retrieval errors and validation results

commit 3

[11/11/2026, 01:22]
decision -> use fastembed and bge embeddings for semantic retrieval independent of lexical retrieval, small model, runs on cpu, uses onnx runtime
alternative -> use only lexical retrieval, use a larger embedding model wih higher resources, use external embedding API (not allowed)
revisit -> evaluate semantic retrieval again after combining it with lexical, maybe consider different embedding model if hybrid retrieval doesn't improve 

[11/11/2026, 02:02]
decision -> saving ICD catalogue embeddings in index file, for faster semantic retrieval across runs
alternative -> no cached embeddings, lead to long embeddings generation time
revisit -> ensure index corresponds to current ICD catalogue nd embedding model

commit 4

[11/11/2026, 02:11]
decision -> combined fts5/bm25 & fastembed ranks using RRF (k = 60), retrieving top 50 candidates
alternative -> use only one retriever or directly combine their raw scores, use other rerankers like weighted RRF, etc.
measurement -> Hit@20: 38/60 = 63.33% which is better than semantic/lexical. hybrid loses to lexical in hit@1,3 but wins over both in hit@5,10,20. I expected this = maximise recall first, then improve ranking later
revisit -> experiment with retrieval limits, RRF params, rank improvement based on error analysis, add other rerankers like weighted RRF & combSUM & combMNZ

[11/11/2026, 03:04]
decision -> retaining RRF with k=60 over weighted RRF & combSUM because we can use LLM to rerank better but missing retrieval will damage more
alternative -> switch to weighted RRF or combSUM
measurement -> RRF achieved Hit@1 16.67%, Hit@3 36.67%, Hit@5 45.00%, Hit@10 55.00%, Hit@20 63.33%, Hit@50 73.33%. weighted RRF achieved Hit@50 58.33%, while CombSUM achieved 71.67%. combSUM had slightly better Hit@20 (65.00% vs 63.33%)
revisit -> reconsider fusion weights/methods if LLM eval shows candidate recall can be improved

commit 5

[11/11/2026, 03:08]
decision -> picked qwen3:1.7b over qwen3:0.6b for LLM reranker, better clinical response
alternative -> sentence transformers, high resource model, low resource model with less accurate response
revisit -> maintain fallback when LLM fails

commit 6

[11/11/2026, 04:24]
decision -> qwen selected broader category instead of more specific one, need to utilize parent-child relation, pass hierarchy info to LLM, but avoid using residual flag as LLM is confusing it with clinical terminology, fixing URI-based hierarchy mapping 
alternative -> 
measurement -> dev-001 is predicated as 1F40 instead of the specific gold code 1F40.Z in this case, hierarchy is faulty as the 1F40.* leafs are not registering 1F40 as its parent
revisit -> incorporate parent hierarchy in candidates retrieval

[11/11/2026, 05:08]
decision -> to implement hierarchy aware reranking, remove non-leaf parent categories
alternative -> repeatedly adding prompt instructions to LLM isn't fixing small model's code-selection errors
revisit -> when done with reranker optimizations





