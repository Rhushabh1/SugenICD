# follows same structure as evaluate_hybrid.py
import json
from pathlib import Path 

from src.retrieval.lexical import build_index as build_lexical_index
from src.retrieval.semantic import build_index as build_semantic_index
from src.retrieval.hybrid import search as hybrid_search
from src.reranking.llm import rerank, filter_candidates


ROOT = Path(__file__).resolve().parent.parent

NOTES_FILE = ROOT / "data" / "dev_notes.jsonl"
MMS_FILE = ROOT / "data" / "mms-2026-01.json"


# loading the first dev note
with open(NOTES_FILE, "r", encoding = "utf-8") as file:
	# for line in file:
	# 	if line.strip():
	# 		notes.append(json.loads(line))
	note = json.loads(file.readline())


# build both retrieval indexes
print("Building lexical index...")
lexical_conn, parents = build_lexical_index(MMS_FILE)

print("Building semantic index...")
model, catalogue, embeddings, semantic_parents = build_semantic_index(MMS_FILE)

print("Lexical & Semantic indexes ready")
# print(f"Parent-child relations: {len(parents)}")
# print(f"Embedding dims: {embeddings.shape}")


# LLM reranking on just 1 note
print("\nRetrieving candidates...")

candidates = hybrid_search(note["text"], lexical_conn, parents, model, catalogue, embeddings, limit = 50, method = "rrf")
print(f"Candidates retrieved: {len(candidates)}")


# inspecting malaria candidates
catalogue_by_code = {entry["code"]: entry for entry in catalogue}

# add catalogue metadata to retrieved candidates
for candidate in candidates:
	entry = catalogue_by_code[candidate["code"]]
	candidate["leaf"] = entry["leaf"]



# inspect whether the gold code was retrieved
gold_codes = set(note["gold"])
candidate_codes = [candidate["code"] for candidate in candidates]

print("\nGold codes:", gold_codes)
print("Gold retrieved:", gold_codes.intersection(candidate_codes))

# inspect the candidates passed through the filter
filtered_candidates = filter_candidates(note["text"], candidates)
filtered_codes = [candidate["code"] for candidate in filtered_candidates]

print("\nGold codes after filtering:", gold_codes.intersection(filtered_codes))

print("\nRelevant malaria candidates:")
for candidate in candidates:
	if candidate["code"].startswith(("1F40", "1F41")):
		print(f"{candidate["code"]} | leaf: {candidate["leaf"]} | {candidate["title"]} | passed filter: {[candidate["code"] in filtered_codes]}")


# run LLM reranker
print("\nRunning qwen3:1.7b reranking...")
result = rerank(note["text"], candidates, catalogue_by_code, model = "qwen3:1.7b")

# show note prediction stats
print(f"\nNote: {note["text"]}")
print(f"\nSelected codes: {result["codes"]}")
print(f"\nReason: {result["reason"]}")
print(f"\nGold codes: {note["gold"]}")

# evaluate this note
gold_codes = set(note["gold"])
pred_codes = result["codes"]

# testing retrieval recall
candidate_codes = [candidate["code"] for candidate in candidates]
print(f"\nCorrect code retrieved: {gold_codes.intersection(candidate_codes)}")

# testing LLM reranker
print(f"\ntop-1: {bool(gold_codes.intersection(pred_codes[:1]))}")
print(f"\ntop-3: {bool(gold_codes.intersection(pred_codes[:3]))}")


lexical_conn.close()