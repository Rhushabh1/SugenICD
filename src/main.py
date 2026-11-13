import os
import json
import sys
import time
from pathlib import Path 

from src.retrieval.lexical import build_index as build_lexical_index
from src.retrieval.semantic import build_index as build_semantic_index
from src.retrieval.hybrid import search as hybrid_search
from src.reranking.llm import rerank, prepare_candidates
from src.reranking.fusion import fuse_candidates
from src.llm.config import get_provider, get_model



# --------------------------------------------------
# Frozen development configuration
# --------------------------------------------------

SEMANTIC_MODE = "title"

RETRIEVAL_METHOD = "weighted_rrf"
RETRIEVAL_LIMIT = 50
RETRIEVAL_LEXICAL_WEIGHT = 0.5
RETRIEVAL_SEMANTIC_WEIGHT = 0.5
RETRIEVAL_RRF_K = 60

FUSION_RETRIEVAL_WEIGHT = 0.3
FUSION_LLM_WEIGHT = 0.7
FUSION_RRF_K = 60

# # Ollama model used during development evaluation.
# LLM_MODEL = "qwen3:1.7b"

# --------------------------------------------------



def save_metrics(path, metrics):
	with open(path, "w", encoding = "utf-8") as file:
		json.dump(metrics, file, indent = 4)


def update_progress(count, total):
	width = 30
	filled = int(width * count / total) if total else width
	bar = "#" * filled + "-" * (width - filled)

	sys.stdout.write(f"\rProgress: [{bar}] {count}/{total}")
	sys.stdout.flush()


# temporary main file to test docker and i/o formats
# normalizing every note but no predicted codes yet
def main():
	if len(sys.argv) != 3:
		print("Usage: python -m src.main input.jsonl output.jsonl")
		sys.exit(1)

	input_file = Path(sys.argv[1])
	output_file = Path(sys.argv[2])
	output_file.parent.mkdir(parents = True, exist_ok = True)

	data_dir = Path(os.getenv("DATA_DIR", "data"))
	# using enriched data for retrieval
	mms_file = data_dir / "mms-2026-01-enriched.json"
	# using official catalogue to validate final codes
	official_mms_file = data_dir / "mms-2026-01.json"

	metrics_file = output_file.parent / "run_metrics.json"
	errors_file = output_file.parent / "errors.jsonl"

	# select provider and model once for the entire run
	provider = get_provider()
	model_name = get_model(provider)

	print("Provider:", provider)
	print("Model:", model_name)

	print("\nFrozen configuration")
	print("--------------------")
	print("Semantic mode:", SEMANTIC_MODE)
	print("Retrieval method:", RETRIEVAL_METHOD)
	print("Retrieval limit:", RETRIEVAL_LIMIT)
	print("Retrieval weights:", RETRIEVAL_LEXICAL_WEIGHT, RETRIEVAL_SEMANTIC_WEIGHT)
	print("Retrieval RRF K:", RETRIEVAL_RRF_K)
	print("Fusion weights:", FUSION_RETRIEVAL_WEIGHT, FUSION_LLM_WEIGHT)
	print("Fusion RRF K:", FUSION_RRF_K)
	print("--------------------\n")

	# count notes for progress bar
	with open(input_file, "r", encoding = "utf-8") as file:
		total = sum(1 for line in file if line.strip())

	# load official valid ICD codes
	with open(official_mms_file, "r", encoding = "utf-8") as file:
		official_catalogue = json.load(file)

	valid_codes = {entry["code"] for entry in official_catalogue}


	# build both retrieval indexes
	print("Building lexical index...")
	lexical_conn, parents = build_lexical_index(mms_file)

	print("Building semantic index...")
	model, catalogue, embeddings, semantic_parents = build_semantic_index(mms_file, semantic_mode=SEMANTIC_MODE)

	catalogue_by_code = {entry["code"]: entry for entry in catalogue}

	print("Retrieval indexes ready\n")

	count = 0
	llm_calls = 0
	llm_errors = 0
	fallback_count = 0
	empty_predictions = 0
	start_time = time.time()

	metrics = {
				"provider": provider,
				"model": model_name,

				"semantic_mode": SEMANTIC_MODE,
				"retrieval_method": RETRIEVAL_METHOD,
				"retrieval_limit": RETRIEVAL_LIMIT,
				"retrieval_lexical_weight": RETRIEVAL_LEXICAL_WEIGHT,
				"retrieval_semantic_weight": RETRIEVAL_SEMANTIC_WEIGHT,
				"retrieval_rrf_k": RETRIEVAL_RRF_K,
				"fusion_retrieval_weight": FUSION_RETRIEVAL_WEIGHT,
				"fusion_llm_weight": FUSION_LLM_WEIGHT,
				"fusion_rrf_k": FUSION_RRF_K,
				
				"total_notes": total,
				"processed_notes": 0,
				"llm_calls": 0,
				"llm_errors": 0,
				"fallback_notes": 0,
				"empty_predictions": 0
				}

	try:
		with open(input_file, "r", encoding = "utf-8") as infile, open(output_file, "w", encoding = "utf-8") as outfile, open(errors_file, "w", encoding = "utf-8") as errorfile:

			# if LLM provider fails once, prevent further API failures
			llm_available = provider != "offline"

			for line in infile:
				if not line.strip():
					continue

				note = json.loads(line)
				note_id = note["id"]

				candidates = []
				filtered_candidates = []
				llm_codes = []

				# ------------------------------------------
				# hybrid retrieval + candidate preparation
				# ------------------------------------------

				try:
					# retrieve candidates using top 30 hybrid search + RRF fusion
					candidates = hybrid_search(note["text"], lexical_conn, parents, model, catalogue, embeddings, limit = RETRIEVAL_LIMIT, method = RETRIEVAL_METHOD, K=RETRIEVAL_RRF_K, lexical_weight=RETRIEVAL_LEXICAL_WEIGHT, semantic_weight=RETRIEVAL_SEMANTIC_WEIGHT)

					# attach leaf metadata & apply filtering
					filtered_candidates = prepare_candidates(note["text"], candidates, catalogue_by_code)

				except Exception as e:
					errorfile.write(json.dumps({
											"id": note_id,
											"stage": "retrieval",
											"error": str(e)
											}) + "\n")
					errorfile.flush()

				# ------------------------------------------
				# LLM reranking
				# ------------------------------------------

				# rerank candidates using LLM provider, if available
				if llm_available and filtered_candidates:
					llm_calls += 1

					try:
						# keep normalized text only for retrieval, not LLM reranker
						result = rerank(note["text"], filtered_candidates, model = model_name, provider = provider)

						llm_codes = result["codes"]

					except Exception as e:
						llm_available = False
						print(f"\nLLM provider failed. Disabling further API calls. Using retrieval fallback for remaining notes.")

						llm_errors += 1
						errorfile.write(json.dumps({
													"id": note_id,
													"stage": "llm",
													"error": str(e)
													}) + "\n")
						errorfile.flush()

				# ------------------------------------------
				# final fusion / fallback
				# ------------------------------------------

				if llm_codes:
					# combine retrieval (0.3) and LLM rankings (0.7)
					fused_codes = fuse_candidates(filtered_candidates, llm_codes, retrieval_weight = FUSION_RETRIEVAL_WEIGHT, llm_weight = FUSION_LLM_WEIGHT, k = FUSION_RRF_K)
				else:
					fused_codes = [
						candidate["code"]
						for candidate in filtered_candidates[:3]
					]
					fallback_count += 1

				# # fill any remaining positions from retrieval
				# for candidate in filtered_candidates:
				# 	code = candidate["code"]

				# 	if len(fused_codes) == 3:
				# 		break

				# 	if code not in fused_codes:
				# 		fused_codes.append(code)

				# ------------------------------------------
				# final validation against official MMS
				# ------------------------------------------

				# (redundant) final validation against the official mms catalogue
				final_codes = []

				for code in fused_codes:
					if code in valid_codes and code not in final_codes:
						final_codes.append(code)

					if len(final_codes) == 3:
						break

				# if not llm_codes:
				# 	fallback_count += 1

				if not final_codes:
					empty_predictions += 1

				# ------------------------------------------
				# write prediction
				# ------------------------------------------

				# final predictions written to outfile immediately
				prediction = {
							"id": note_id,
							"codes": final_codes
							}

				outfile.write(json.dumps(prediction, ensure_ascii = False) + "\n")
				outfile.flush()

				count+=1

				# save metrics at the end
				metrics["processed_notes"] = count
				metrics["llm_calls"] = llm_calls
				metrics["llm_errors"] = llm_errors
				metrics["fallback_notes"] = fallback_count
				metrics["empty_predictions"] = empty_predictions

				# progress bar update
				update_progress(count, total)

	finally:
		lexical_conn.close()

	elapsed = time.time() - start_time

	metrics["elapsed_seconds"] = round(elapsed, 2)
	metrics["average_seconds_per_note"] = round(elapsed / count, 2) if count else 0

	save_metrics(metrics_file, metrics)

	print("\n\nRun completed")
	print("---------------------------------")
	print("Processed notes:	", count)
	print("LLM calls:		", llm_calls)
	print("LLM errors:		", llm_errors)
	print("Fallback notes:		", fallback_count)
	print("Empty predictions:	", empty_predictions)
	print("Elapsed time:		", round(elapsed, 2), "sec")
	print("---------------------------------")
	print("Predictions:	", output_file)
	print("Metrics:	", metrics_file)
	print("Errors:		", errors_file)


if __name__ == "__main__":
	main()