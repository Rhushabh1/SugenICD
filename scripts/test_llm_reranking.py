# follows same structure as evaluate_hybrid.py
import json
from pathlib import Path 

from src.retrieval.lexical import build_index as build_lexical_index
from src.retrieval.semantic import build_index as build_semantic_index
from src.retrieval.hybrid import search as hybrid_search
from src.reranking.llm import rerank


ROOT = Path(__file__).resolve().parent.parent

NOTES_FILE = ROOT / "data" / "dev_notes.jsonl"
MMS_FILE = ROOT / "data" / "mms-2026-01.json"


# loading the dev notes
# notes = []
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

# run LLM reranker
print("\nRunning qwen3:1.7b reranking...")
result = rerank(note["text"], candidates)

# show note prediction stats
print(f"\nNote: {note["text"]}")
print(f"\nSelected codes: {result["codes"]}")
print(f"\nReason: {result["reason"]}")
print(f"\nGold codes: {note["gold"]}")

# testing retrieval recall
candidate_codes = [candidate["code"] for candidate in candidates]
print(f"\nCorrect code retrieved: {set(note["gold"]).intersection(candidate_codes)}")

# testing LLM reranker
print(f"\ntop-1: {bool(set(note["gold"]).intersection(result["codes"][:1]))}")
print(f"\ntop-3: {bool(set(note["gold"]).intersection(result["codes"][:3]))}")


lexical_conn.close()