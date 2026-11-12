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

[11/11/2026, 19:17]
decision -> compare top 3 filtered RRF candidates against qwen's reranked top 3 using identical candidates for both, comparing it with fallback routine to create top 3 answers
measurement -> record retrieval hit@k, final top1/top3, official dev score and LLM errors across all 60 notes. hybrid retrieval scored 20.4/75, LLM reranking scored 31.2/75, and LLM with hybrid fallback scored 32.9/75. top1 accuracy improved from 21.67% to 40.00%.
alternative -> blindly enrich and add features without testing hybrid, LLM, fallback and how they contribute to the final predictions
revisit -> use it to benchmark after enriching ICD data

commit 7

[11/11/2026, 20:47]
decision -> using a shared candidate preparation function to attach ICD catalogue metadata and apply leaf node and keyword filtering before LLM, retrieval handle normalization
alternative -> perform metadata enrichment, normalization, filtering independently, keep filtering inside LLM reranker
revisit -> if additional filtering or metadata introduced

[11/11/2026, 20:53]
decision -> constrain LLM output to candidate codes only (no hallucinations or invalid codes), improve prompt's specificity rules (wrong sibling and overly specific codes), no reason needed (shorten the llm time)
measurement -> current LLM achieved 40.00% top1 and 55.00% top3 accuracy, scoring 33.8/75. with hybrid fallback, the score increased to 34.2/75 and top3 accuracy to 56.67%. no LLM errors occurred. this is the best result among the tested configs
revisit -> improve candidate recall and clinical specificity using enriched WHO terms, then evaluate retrieval and reranking independently

[11/11/2026, 22:26]
decision -> enrich ICD MMS catalogue using WHO data, including terms and definitions
alternative -> continue retrieving using category titles alone
revisit -> evaluate improvement in retrieval hit@k and final top1/top3 accuracy

commit 8

[11/11/2026, 00:40]
decision -> enriched json is tested for lexical analysis against the hybrid baseline of hit@50 = 73.33%
measurement -> enriched lexical index achieved Hit@1 35.00%, Hit@3 51.67%, Hit@5 53.33%, Hit@10 56.67% and Hit@20 66.67% on 60 notes. compared with the original catalogue, Hit@20 improved from 51.67% to 66.67%

[11/11/2026, 01:14]
decision -> tested hybrid retrieval with enriched mms file. RRF remains strong across hit@3,10,20 whereas combSUM and weighted RRF lose more and gain marginally
measurement -> enriched RRF achieved Hit@1 28.33%, Hit@3 50.00%, Hit@5 56.67%, Hit@10 70.00%, Hit@20 78.33% and Hit@50 88.33% on 60 development notes. compared with the original RRF baseline, Hit@50 improved from 73.33% to 88.33%. RRF achieved Hit@10 70.00%, Hit@20 78.33% and Hit@50 88.33%. combSUM achieved slightly higher Hit@50 at 90.00%, while weighted RRF achieved the best Hit@1 at 31.67%

[11/11/2026, 01:48]
decision -> not passing enriched metadata to LLM
measurement -> the accuracy collapsed from 30% LLM misses to 100% LLM misses after passing WHO metadata to it
revisit -> when noisy associations won't hurt LLM reranking

[11/11/2026, 02:14]
decision -> adopting weighted RRF (0.3 retrieval + 0.7 LLM, k = 60) as experimental ranking strategy
measurement -> top1 improved from 61 to 69% and top3 remained unchanged for a small test sample of 13 notes
alternative -> use LLM ranking alone or retain existing operational fallback strategy (trusting LLM too much and filling gaps with retrieval)
revisit -> validate once testing it on all 60 notes before finalizing the weights

[11/11/2026, 03:07]
decision -> reduce hybrid candidates from 50 to 30 and compact the LLM prompt
measurement -> processing time decreased from 25-30sec to under 10sec per note, fused score reached 35/75 (highest yet)
issue -> 10/60 LLM responses failed json prsing after reducing num_predict to 25
revisit -> increase generation tokens to 40 and validate parsing reliability

[11/11/2026, 03:24]
decision -> freezing current 30-candidate hybrid retrieval + qwen3:1.7b reranking + weighted RRF (0.3 retrieval + 0.7 LLM) for now and working on final submission format and API keys
measurement -> new best fused score of 39.6/75, no LLM errors
alternative -> keep benchmarking and experimenting with strategies without a proper final submission package handy
revisit -> once the final submission package is working and able to reproduce the same results in docker commands

[11/11/2026, 03:29]
decision -> provider behaviour -> (auto, ollama, openai, anthropic, offline) to set up easy test deployment and development env
alternative -> manually change variables to switch from one mode to another
revisit -> no (best portable option)

[11/11/2026, 04:00]
decision -> added openai & anthropic message PAI with structured json output, sharing existing prompt and candidate validation with ollama
alternative -> use prompt-only json generation or maintain separate reranking logic for each provider
revisit -> compare accuracy, latency and API cost on same dev notes when grading compatible API cred are available

[11/11/2026, 04:32]
decision -> selected gpt-6-luna and claude-haiku-5-5 as low-cost API model candidates for testing, with one provider selected per run
alternative -> use more expensive models or call both providers for every note
measurement -> current published standard rates show USD 0.1/M input and 0.5/M output tokens, actual workload cost may differ

[11/11/2026, 06:01]
decision -> verified docker execution with ollama through docker internal
measurement -> 60/60 notes processed, 0 llm errors, 0 fallbacks, 0 invalid codes, score 40.4/75 by score.py, runtime = 560sec
alternative -> run ollama inside docker or use API providers for local testing
revisit -> compare API provider accuracy, latency, and cost when grading creds available, if any

commit 9


