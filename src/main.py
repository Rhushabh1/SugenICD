import json
import sys
from pathlib import Path 

from src.preprocessing.normalize import normalize_text
from src.retrieval.lexical import build_index as build_lexical_index
from src.retrieval.semantic import build_index as build_semantic_index
from src.retrieval.hybrid import search as hybrid_search
from src.reranking.llm import rerank


# temporary main file to test docker and i/o formats
# normalizing every note but no predicted codes yet
def main():
	if len(sys.argv) != 3:
		print("Usage: python -m src.main input.jsonl output.jsonl")
		sys.exit(1)

	input_file = Path(sys.argv[1])
	output_file = Path(sys.argv[2])
	mms_file = Path("data/mms-2026-01.json")

	output_file.parent.mkdir(parents = True, exist_ok = True)

	# build both retrieval indexes
	print("Building lexical index...")
	lexical_conn, parents = build_lexical_index(mms_file)

	print("Building semantic index...")
	model, catalogue, embeddings, semantic_parents = build_semantic_index(mms_file)

	catalogue_by_code = {entry["code"]: entry for entry in catalogue}

	print("Lexical & Semantic indexes ready")

	count = 0

	with open(input_file, "r", encoding = "utf-8") as infile, open(output_file, "w", encoding = "utf-8") as outfile:
		for line in infile:
			if not line.strip():
				continue

			note = json.loads(line)
			print(f"Processing note ID: {note["id"]}")

			# normalize note for retrieval
			normalized_text = normalize_text(note["text"])
			
			# retrieve candidates using hybrid search + RRF fusion
			candidates = hybrid_search(normalized_text, lexical_conn, parents, model, catalogue, embeddings, limit = 50, method = "rrf")

			# rerank candidates using LLM
			codes = []

			try:
				# keep normalized text only for retrieval, not LLM reranker
				result = rerank(note["text"], candidates, catalogue_by_code, model = "qwen3:1.7b")

				codes = result["codes"]

				print("\nCandidates sent to LLM:")
				print(result["candidate_codes"])

				print("\nRaw LLM codes:")
				print(result["raw_codes"])

				print("\nInvalid LLM codes:")
				print(result["invalid_codes"])

				print("\nLLM explanation:")
				print(result["reason"])

				print("\nRaw LLM response:")
				print(result["raw_response"])

				print("\nCandidates sent to LLM:")
				for candidate in result["candidate_details"]:
					print(candidate["code"], "|", candidate["title"])

			except Exception as e:
				print(f"LLM error for {note["id"]}: {str(e)}")

			# final predictions written to outfile
			prediction = {
						"id": note["id"],
						"codes": codes
						}

			outfile.write(json.dumps(prediction, ensure_ascii = False) + "\n")
			outfile.flush()

			count+=1
			print(f"Processed: {count} - {note["id"]}")

	print(f"Processed notes: {count}")
	print(f"Predictions: {output_file}")

	lexical_conn.close()



if __name__ == "__main__":
	main()