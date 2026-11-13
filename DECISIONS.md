# - **Decision**S

## 10/11/2026

#### [20:23]
- **Decision** -> data inspection + score.py + use supplied data & mms catalogue, using them as starting point
- **Alternative** -> start preprocessing/retrieval without a validation script or knowledge of data
- **Revisit** -> check catalogue hierarchy and child-parent relationships when implementing lexical

#### Commit 1

#### [21:49]
- **Decision** -> docker setup with i/o to avoid issues later, keeping it light in dependencies
- **Alternative** -> use larger ML image or install all planned AI dependencies upfront
- **Revisit** -> add dependencies only when implementing retrieval/models

#### [22:07]
- **Decision** -> using python 3.11-slim inside docker while retaining 3.13 for local dev
- **Alternative** -> same python version locally and inside docker
- **Revisit** -> if a required dependency needs a diff python version

#### [22:34]
- **Decision** -> apply norm, lowercase & stripping, whitespace cleanups
- **Alternative** -> remove stop words, punctuations, other tokens
- **Revisit** -> evaluate more normalizations against retrieval accuracy later

#### Commit 2

#### [23:38]
- **Decision** -> use sqlite fts5 with bm25 for lexical retrieval of ICD category candidates from clinical note text. prioritize exact category-title match and derive parent relationships from catalogue uri. also added hit@k metric to establish a baseline first
- **Alternative** -> keyword matching only or introduce elasticsearch or another external search tool -> extra lib dependency
- **Revisit** -> analyse missed gold codes and improve retrieval before adding semantic retrieval

## 11/11/2026

#### [00:53]
- **Decision** -> no truncation of tokens to top 50 & manually expanding clinical & anatomical synonyms for better hit
- **Measurement** -> Hit@20: 25/60 = 41.67% improved to now Hit@20: 31/60 = 51.67%
- **Alternative** -> use only original note tokens, introduces larger external medical terminology database
- **Revisit** -> expand dictionary based on observed retrieval errors and validation results

#### Commit 3

#### [01:22]
- **Decision** -> use fastembed and bge embeddings for semantic retrieval independent of lexical retrieval, small model, runs on cpu, uses onnx runtime
- **Alternative** -> use only lexical retrieval, use a larger embedding model wih higher resources, use external embedding API (not allowed)
- **Revisit** -> evaluate semantic retrieval again after combining it with lexical, maybe consider different embedding model if hybrid retrieval doesn't improve 

#### [02:02]
- **Decision** -> saving ICD catalogue embeddings in index file, for faster semantic retrieval across runs
- **Alternative** -> no cached embeddings, lead to long embeddings generation time
- **Revisit** -> ensure index corresponds to current ICD catalogue nd embedding model

#### Commit 4

#### [02:11]
- **Decision** -> combined fts5/bm25 & fastembed ranks using RRF (k = 60), retrieving top 50 candidates
- **Alternative** -> use only one retriever or directly combine their raw scores, use other rerankers like weighted RRF, etc.
- **Measurement** -> Hit@20: 38/60 = 63.33% which is better than semantic/lexical. hybrid loses to lexical in hit@1,3 but wins over both in hit@5,10,20. I expected this = maximise recall first, then improve ranking later
- **Revisit** -> experiment with retrieval limits, RRF params, rank improvement based on error analysis, add other rerankers like weighted RRF & combSUM & combMNZ

#### [03:04]
- **Decision** -> retaining RRF with k=60 over weighted RRF & combSUM because we can use LLM to rerank better but missing retrieval will damage more
- **Alternative** -> switch to weighted RRF or combSUM
- **Measurement** -> RRF achieved Hit@1 16.67%, Hit@3 36.67%, Hit@5 45.00%, Hit@10 55.00%, Hit@20 63.33%, Hit@50 73.33%. weighted RRF achieved Hit@50 58.33%, while CombSUM achieved 71.67%. combSUM had slightly better Hit@20 (65.00% vs 63.33%)
- **Revisit** -> reconsider fusion weights/methods if LLM eval shows candidate recall can be improved

#### Commit 5

#### [03:08]
- **Decision** -> picked qwen3:1.7b over qwen3:0.6b for LLM reranker, better clinical response
- **Alternative** -> sentence transformers, high resource model, low resource model with less accurate response
- **Revisit** -> maintain fallback when LLM fails

#### Commit 6

#### [04:24]
- **Decision** -> qwen selected broader category instead of more specific one, need to utilize parent-child relation, pass hierarchy info to LLM, but avoid using residual flag as LLM is confusing it with clinical terminology, fixing URI-based hierarchy mapping 
- **Alternative** -> 
- **Measurement** -> dev-001 is predicated as 1F40 instead of the specific gold code 1F40.Z in this case, hierarchy is faulty as the 1F40.* leafs are not registering 1F40 as its parent
- **Revisit** -> incorporate parent hierarchy in candidates retrieval

#### [05:08]
- **Decision** -> to implement hierarchy aware reranking, remove non-leaf parent categories
- **Alternative** -> repeatedly adding prompt instructions to LLM isn't fixing small model's code-selection errors
- **Revisit** -> when done with reranker optimizations

#### [19:17]
- **Decision** -> compare top 3 filtered RRF candidates against qwen's reranked top 3 using identical candidates for both, comparing it with fallback routine to create top 3 answers
- **Measurement** -> record retrieval hit@k, final top1/top3, official dev score and LLM errors across all 60 notes. hybrid retrieval scored 20.4/75, LLM reranking scored 31.2/75, and LLM with hybrid fallback scored 32.9/75. top1 accuracy improved from 21.67% to 40.00%.
- **Alternative** -> blindly enrich and add features without testing hybrid, LLM, fallback and how they contribute to the final predictions
- **Revisit** -> use it to benchmark after enriching ICD data

#### Commit 7

#### [20:47]
- **Decision** -> using a shared candidate preparation function to attach ICD catalogue metadata and apply leaf node and keyword filtering before LLM, retrieval handle normalization
- **Alternative** -> perform metadata enrichment, normalization, filtering independently, keep filtering inside LLM reranker
- **Revisit** -> if additional filtering or metadata introduced

#### [20:53]
- **Decision** -> constrain LLM output to candidate codes only (no hallucinations or invalid codes), improve prompt's specificity rules (wrong sibling and overly specific codes), no reason needed (shorten the llm time)
- **Measurement** -> current LLM achieved 40.00% top1 and 55.00% top3 accuracy, scoring 33.8/75. with hybrid fallback, the score increased to 34.2/75 and top3 accuracy to 56.67%. no LLM errors occurred. this is the best result among the tested configs
- **Revisit** -> improve candidate recall and clinical specificity using enriched WHO terms, then evaluate retrieval and reranking independently

#### [22:26]
- **Decision** -> enrich ICD MMS catalogue using WHO data, including terms and definitions
- **Alternative** -> continue retrieving using category titles alone
- **Revisit** -> evaluate improvement in retrieval hit@k and final top1/top3 accuracy

#### Commit 8

## 12/11/2026

#### [00:40]
- **Decision** -> enriched json is tested for lexical analysis against the hybrid baseline of hit@50 = 73.33%
- **Measurement** -> enriched lexical index achieved Hit@1 35.00%, Hit@3 51.67%, Hit@5 53.33%, Hit@10 56.67% and Hit@20 66.67% on 60 notes. compared with the original catalogue, Hit@20 improved from 51.67% to 66.67%

#### [01:14]
- **Decision** -> tested hybrid retrieval with enriched mms file. RRF remains strong across hit@3,10,20 whereas combSUM and weighted RRF lose more and gain marginally
- **Measurement** -> enriched RRF achieved Hit@1 28.33%, Hit@3 50.00%, Hit@5 56.67%, Hit@10 70.00%, Hit@20 78.33% and Hit@50 88.33% on 60 development notes. compared with the original RRF baseline, Hit@50 improved from 73.33% to 88.33%. RRF achieved Hit@10 70.00%, Hit@20 78.33% and Hit@50 88.33%. combSUM achieved slightly higher Hit@50 at 90.00%, while weighted RRF achieved the best Hit@1 at 31.67%

#### [01:48]
- **Decision** -> not passing enriched metadata to LLM
- **Measurement** -> the accuracy collapsed from 30% LLM misses to 100% LLM misses after passing WHO metadata to it
- **Revisit** -> when noisy associations won't hurt LLM reranking

#### [02:14]
- **Decision** -> adopting weighted RRF (0.3 retrieval + 0.7 LLM, k = 60) as experimental ranking strategy
- **Measurement** -> top1 improved from 61 to 69% and top3 remained unchanged for a small test sample of 13 notes
- **Alternative** -> use LLM ranking alone or retain existing operational fallback strategy (trusting LLM too much and filling gaps with retrieval)
- **Revisit** -> validate once testing it on all 60 notes before finalizing the weights

#### [03:07]
- **Decision** -> reduce hybrid candidates from 50 to 30 and compact the LLM prompt
- **Measurement** -> processing time decreased from 25-30sec to under 10sec per note, fused score reached 35/75 (highest yet)
issue -> 10/60 LLM responses failed json prsing after reducing num_predict to 25
- **Revisit** -> increase generation tokens to 40 and validate parsing reliability

#### [03:24]
- **Decision** -> freezing current 30-candidate hybrid retrieval + qwen3:1.7b reranking + weighted RRF (0.3 retrieval + 0.7 LLM) for now and working on final submission format and API keys
- **Measurement** -> new best fused score of 39.6/75, no LLM errors
- **Alternative** -> keep benchmarking and experimenting with strategies without a proper final submission package handy
- **Revisit** -> once the final submission package is working and able to reproduce the same results in docker commands

#### [03:29]
- **Decision** -> provider behaviour -> (auto, ollama, openai, anthropic, offline) to set up easy test deployment and development env
- **Alternative** -> manually change variables to switch from one mode to another
- **Revisit** -> no (best portable option)

#### [04:00]
- **Decision** -> added openai & anthropic message PAI with structured json output, sharing existing prompt and candidate validation with ollama
- **Alternative** -> use prompt-only json generation or maintain separate reranking logic for each provider
- **Revisit** -> compare accuracy, latency and API cost on same dev notes when grading compatible API cred are available

#### [04:32]
- **Decision** -> selected gpt-6-luna and claude-haiku-5-5 as low-cost API model candidates for testing, with one provider selected per run
- **Alternative** -> use more expensive models or call both providers for every note
- **Measurement** -> current published standard rates show USD 0.1/M input and 0.5/M output tokens, actual workload cost may differ

#### [06:01]
- **Decision** -> verified docker execution with ollama through docker internal
- **Measurement** -> 60/60 notes processed, 0 llm errors, 0 fallbacks, 0 invalid codes, score 40.4/75 by score.py, runtime = 560sec
- **Alternative** -> run ollama inside docker or use API providers for local testing
- **Revisit** -> compare API provider accuracy, latency, and cost when grading creds available, if any

#### Commit 9

#### [08:52]
- **Decision** -> rewriting evaluate_llm.py & evaluate_fusion.py for easy reproducible dev testing using a single file as store of predictions

#### [12:24]
- **Decision** -> forward https_proxy into docker container and retain urllib for API requests, disable further LLM requess after first provider failure and continue using retrieval candidates

#### [13:54]
- **Decision** -> finalizing (40.42/75 best observed dev score), semantic mode - title, hybrid retrieval (weighted_rrf, lex 0.5, semantic 0.5, K 60), candidate limit 50, fusion
(retrieval 0.3, LLM 0.7, K 30)
- **Measurement** -> evaluate_llm and evaluate_fusion gave the best score in the above config
- **Revisit** -> fine tune later if time permits

#### [15:06]
- **Decision** -> added rigorous docker image tests, provider tests, testing it under different grading scenarios and verifying http proxy and env vars and API keys pass through to the docker image
- **Alternative** -> assume API path works because ollama works in docker or assume mocked HTTP tests prove that readl creds and proxy access work

#### Commit 10

