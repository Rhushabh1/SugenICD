# follows same structure as evaluate_lexical.py
import json
from pathlib import Path 

from src.retrieval.semantic import build_index, search


ROOT = Path(__file__).resolve().parent.parent

NOTES_FILE = ROOT / "data" / "dev_notes.jsonl"
MMS_FILE = ROOT / "data" / "mms-2026-01.json"


# loading the dev notes
notes = []
with open(NOTES_FILE, "r", encoding = "utf-8") as file:
	for line in file:
		if line.strip():
			notes.append(json.loads(line))

# building semantic catalogue search index
print("Building semantic index...")
model, catalogue, embeddings, parents = build_index(MMS_FILE)

print("Catalogue indexed")
print(f"Parent-child relations: {len(parents)}")
print(f"Embedding dims: {embeddings.shape}")

# HIT@K -> evaluate whether gold code appears in top k results or not
hits = {1: 0,
		3: 0,
		5: 0,
		10: 0,
		20: 0}

for note in notes:
	results = search(note["text"], model, catalogue, embeddings, parents, limit = 20)

	# predicted values
	predicted_codes = [result["code"] for result in results]

	# truth values
	gold_codes = set(note["gold"])

	# to check for any hit within k results
	for k in hits:
		if gold_codes.intersection(predicted_codes[:k]):
			hits[k] += 1

print("\nSemantic retrieval results ->")
print(f"Total notes: {len(notes)}")

for k, count in hits.items():
	precent = (count / len(notes)) * 100
	print(f"Hit@{k}: {count}/{len(notes)} = {precent:.2f}%")


# to show few examples for manual inspection
print("\nSample retrieval results ->")

for note in notes[:5]:
	results = search(note["text"], model, catalogue, embeddings, parents, limit = 5)

	# predicted values
	predicted_codes = [result["code"] for result in results]

	# truth values
	gold_codes = set(note["gold"])

	print(f"\nNote: {note["id"]}")
	print(f"Gold: {gold_codes}")
	print(f"Retrieval codes: {predicted_codes}")

	for result in results:
		print(f"{result["code"]} | {result["title"]} | similarity: {result["score"]:.4f} | parent: {result["parent_code"]}")

