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

[10/11/2026, 00:53]
decision -> no truncation of tokens to top 50 & manually expanding clinical & anatomical synonyms for better hit
measurement -> Hit@20: 25/60 = 41.67% improved to now Hit@20: 31/60 = 51.67%
alternative -> use only original note tokens, introduces larger external medical terminology database
revisit -> expand dictionary based on observed retrieval errors and validation results

commit 3

[10/11/2026, 01:22]
decision -> use fastembed and bge embeddings for semantic retrieval independent of lexical retrieval, small model, runs on cpu, uses onnx runtime
alternative -> use only lexical retrieval, use a larger embedding model wih higher resources, use external embedding API (not allowed)
revisit -> evaluate semantic retrieval again after combining it with lexical, maybe consider different embedding model if hybrid retrieval doesn't improve 

[10/11/2026, 02:02]
decision -> saving ICD catalogue embeddings in index file, for faster semantic retrieval across runs
alternative -> no cached embeddings, lead to long embeddings generation time
revisit -> ensure index corresponds to current ICD catalogue nd embedding model

commit 4


