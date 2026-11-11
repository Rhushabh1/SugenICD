# follows same structure as evaluate_semantic.py
import json
from pathlib import Path 

from src.retrieval.lexical import build_index as build_lexical_index
from src.retrieval.semantic import build_index as build_semantic_index
from src.retrieval.hybrid import search


ROOT = Path(__file__).resolve().parent.parent

NOTES_FILE = ROOT / "data" / "dev_notes.jsonl"
MMS_FILE = ROOT / "data" / "mms-2026-01-enriched.json"


# loading the dev notes
notes = []
with open(NOTES_FILE, "r", encoding = "utf-8") as file:
	for line in file:
		if line.strip():
			notes.append(json.loads(line))

# build both retrieval indexes
print("Building lexical index...")
lexical_conn, parents = build_lexical_index(MMS_FILE)

print("Building semantic index...")
model, catalogue, embeddings, semantic_parents = build_semantic_index(MMS_FILE)

print("Lexical & Semantic indexes ready")
# print(f"Parent-child relations: {len(parents)}")
# print(f"Embedding dims: {embeddings.shape}")

# for benchmarking different fusion methods
METHODS = ["rrf", "weighted_rrf", "combsum"]
# HIT@K -> evaluate whether gold code appears in top k results or not
K_VALUES = [1, 3, 5, 10, 20, 50]



# evaluate each fusion method
for method in METHODS:
	hits = {k: 0 for k in K_VALUES}

	for note in notes:
		results = search(note["text"], lexical_conn, parents, model, catalogue, embeddings, limit = 50, method = method)

		# predicted values
		predicted_codes = [result["code"] for result in results]

		# truth values
		gold_codes = set(note["gold"])

		# to check for any hit within k results
		for k in K_VALUES:
			if gold_codes.intersection(predicted_codes[:k]):
				hits[k] += 1

	print(f"\nHybrid + {method.upper()} results ->")
	print(f"Total notes: {len(notes)}")

	for k, count in hits.items():
		precent = (count / len(notes)) * 100
		print(f"Hit@{k}: {count}/{len(notes)} = {precent:.2f}%")



# to show few examples for manual inspection
print("\nSample retrieval results ->")

for method in METHODS:
	print(f"\n{method.upper()} ->")

	for note in notes[:3]:
		results = search(note["text"], lexical_conn, parents, model, catalogue, embeddings, limit = 5, method = method)

		# predicted values
		predicted_codes = [result["code"] for result in results]

		# truth values
		gold_codes = set(note["gold"])

		print(f"\nNote: {note["id"]}")
		print(f"Gold: {gold_codes}")
		print(f"Retrieval codes: {predicted_codes}")

		# for result in results:
		# 	print(f"{result["code"]} | {result["title"]} | lexical rank: {result["lexical_rank"]} | semantic rank: {result["semantic_rank"]} | Score: {result["combined_score"]:.4f}")


lexical_conn.close()