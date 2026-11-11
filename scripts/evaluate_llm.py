import json
from pathlib import Path

from src.retrieval.lexical import build_index as build_lexical_index
from src.retrieval.semantic import build_index as build_semantic_index
from src.retrieval.hybrid import search as hybrid_search
from src.reranking.llm import rerank, prepare_candidates
from src.reranking.fusion import fuse_candidates


# calculate hit@k for the given results & truth values = gold_notes
def calculate_metrics(results, gold_notes):
	total = len(results)
	metrics = {}

	# for each K value
	for k in [1, 3, 5, 10, 20, 50]:
		hit = 0

		for result in results:
			# truth value
			gold = set(gold_notes[result["id"]])
			# top k predicted values
			predicted = result["codes"][:k]

			if any(code in gold for code in predicted):
				hit += 1

		# save hit@k in the metrics
		metrics[f"hit@{k}"] = round((hit / total) * 100, 2)

	return metrics


# main benchmarking method for hybrid & hybrid + LLM
def main():
	# input files extracted
	data_file = Path("data/dev_notes.jsonl")
	mms_file = Path("data/mms-2026-01-enriched.json")
	
	# to validate output directory
	output_dir = Path("out")
	output_dir.mkdir(exist_ok=True)

	# extract notes for prediction
	with open(data_file, "r", encoding="utf-8") as file:
		notes = [json.loads(line) for line in file if line.strip()]

	# truth values
	gold_notes = {note["id"]: note["gold"] for note in notes}

	print("Building lexical index...")
	lexical_conn, parents = build_lexical_index(mms_file)

	print("Building semantic index...")
	model, catalogue, embeddings, semantic_parents = build_semantic_index(mms_file)

	catalogue_by_code = {entry["code"]: entry for entry in catalogue}

	print("Indexes are ready")
	print(f"Total notes: {len(notes)}")

	# saving hybrid candidates & preds
	retrieval_results = []
	hybrid_results = []
	filtered_results = []

	# saving LLM preds, LLM fallback preds, LLM errors
	llm_results = []
	fallback_results = []
	llm_errors = []
	fused_results = []

	# saved in following output files
	hybrid_file = open(output_dir / "3_hybrid_predictions.jsonl", "w", encoding = "utf-8")
	llm_file = open(output_dir / "3_llm_predictions.jsonl", "w", encoding = "utf-8")
	fallback_file = open(output_dir / "3_llm_fallback_predictions.jsonl", "w", encoding = "utf-8")
	details_file = open(output_dir / "3_llm_details.jsonl", "w", encoding = "utf-8")
	errors_file = open(output_dir / "3_llm_errors.jsonl", "w", encoding = "utf-8")
	# weighted RRF to replace fallback strategy
	fused_file = open(output_dir / "3_llm_fused_predictions.jsonl", "w", encoding="utf-8")

	try:
		for i, note in enumerate(notes, start = 1):
			note_id = note["id"]
			original_text = note["text"]

			# show progress log
			print(f"\nProcessing {i}/{len(notes)}: {note_id}")

			# retrieve the top 50 hybrid candidates
			candidates = hybrid_search(original_text, lexical_conn, parents, model, catalogue, embeddings, limit=30, method="rrf")

			retrieval_results.append({
									"id": note_id,
									"codes": [candidate["code"] for candidate in candidates]
									})

			# filtered right before LLM reranker is called (filtering out non-leaf codes and specific keywords)
			filtered_candidates = prepare_candidates(original_text, candidates, catalogue_by_code)

			# baseline preds -> first three filtered candidates
			hybrid_codes = [candidate["code"] for candidate in filtered_candidates[:3]]

			hybrid_results.append({
								"id": note_id,
								"codes": hybrid_codes
								})

			filtered_results.append({
									"id": note_id,
									"codes": [candidate["code"] for candidate in filtered_candidates]
									})

			# initializing before try block
			llm_codes = []
			# reason = ""
			raw_response = ""
			raw_codes = []
			invalid_codes = []
			fused_codes = []
			error = ""

			try:
				# keep normalized text only for retrieval, not LLM reranker
				result = rerank(original_text, filtered_candidates, model="qwen3:1.7b")

				llm_codes = result["codes"]
				# reason = result["reason"]
				raw_response = result["raw_response"]
				raw_codes = result["raw_codes"]
				invalid_codes = result["invalid_codes"]
				fused_codes = fuse_candidates(filtered_candidates, llm_codes, retrieval_weight=0.3, llm_weight=0.7, k=60)

				if not llm_codes:
					error = "LLM returned no valid codes"

			except Exception as e:
				error = str(e)
				print(f"LLM error: {error}")
				fused_codes = [candidate["code"] for candidate in filtered_candidates[:3]]

			# pure LLM result, without fallback
			llm_results.append({
								"id": note_id,
								"codes": llm_codes
								})

			fused_results.append({
								"id": note_id,
								"codes": fused_codes
								})

			# proper operational fallback ->
			# use hybrid candidates if LLM returns no valid codes
			fallback_codes = list(llm_codes)
			# if LLM returns fewer than three, fill remaining positions with hybrid candidates not already selected (optional, to increase hit rate)
			for code in hybrid_codes:
				if len(fallback_codes) == 3:
					break

				if code not in fallback_codes:
					fallback_codes.append(code)

			fallback_results.append({
									"id": note_id,
									"codes": fallback_codes
									})

			# save all predictions/results immediately in the loop
			hybrid_file.write(json.dumps(hybrid_results[-1]) + "\n")
			llm_file.write(json.dumps(llm_results[-1]) + "\n")
			fallback_file.write(json.dumps(fallback_results[-1]) + "\n")
			fused_file.write(json.dumps(fused_results[-1], ensure_ascii = False) + "\n")

			# save candidate details and the raw LLM response (in proper execution sequence)
			details = {
						"id": note_id,
						"candidates": [{
										"code": candidate["code"],
										"title": candidate["title"]
										} for candidate in filtered_candidates],
						"hybrid_codes": hybrid_codes,
						"llm_codes": llm_codes,
						"raw_codes": raw_codes,
						"invalid_codes": invalid_codes,
						# "reason": reason,
						"raw_response": raw_response
						}

			details_file.write(json.dumps(details, ensure_ascii=False) + "\n")

			# to log errors separately
			if error:
				error_record = {
								"id": note_id,
								"error": error,
								"raw_response": raw_response
								}

				llm_errors.append(error_record)
				errors_file.write(json.dumps(error_record, ensure_ascii=False) + "\n")

			# flush after every note so partial results are preserved
			hybrid_file.flush()
			llm_file.flush()
			fallback_file.flush()
			details_file.flush()
			errors_file.flush()
			fused_file.flush()

			# for CLI & logging
			print("Hybrid: ", hybrid_codes)
			print("LLM: ", llm_codes)
			print("Fallback: ", fallback_codes)
			print("Fused: ", fused_codes)

	finally:
		# closing all files and connections
		hybrid_file.close()
		llm_file.close()
		fallback_file.close()
		details_file.close()
		errors_file.close()
		fused_file.close()
		lexical_conn.close()

	# calculate metrics using the results already collected
	metrics = {
				"hybrid_retrieval": calculate_metrics(retrieval_results, gold_notes),
				"filtered_retrieval": calculate_metrics(filtered_results, gold_notes),
				"hybrid_top_3": calculate_metrics(hybrid_results, gold_notes),
				"llm_only": calculate_metrics(llm_results, gold_notes),
				"llm_with_fallback": calculate_metrics(fallback_results, gold_notes),
				"llm_fused": calculate_metrics(fused_results, gold_notes),
				"llm_error_count": len(llm_errors),
				"total_notes": len(notes)
				}

	# final benchmark file
	with open(output_dir / "3_benchmark_results.json", "w", encoding = "utf-8") as file:
		json.dump(metrics, file, indent=4)

	print("\nBenchmark completed")
	# print(json.dumps(metrics, indent=4))

	print("\nOutput files saved in: ", output_dir)


if __name__ == "__main__":
	main()
