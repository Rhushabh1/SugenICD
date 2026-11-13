import json
import argparse
from pathlib import Path

from src.retrieval.lexical import build_index as build_lexical_index
from src.retrieval.lexical import search as lexical_search
from src.retrieval.semantic import build_index as build_semantic_index
from src.retrieval.semantic import search as semantic_search
from src.retrieval.hybrid import search as hybrid_search
from src.reranking.llm import rerank, prepare_candidates
from src.reranking.fusion import fuse_candidates


ROOT = Path(__file__).resolve().parent.parent

NOTES_FILE = ROOT / "data" / "dev_notes.jsonl"
MMS_FILE = ROOT / "data" / "mms-2026-01-enriched.json"
OUTPUT_DIR = ROOT / "out"
# EVALUATION_FILE = OUTPUT_DIR / "evaluation.jsonl"

RETRIEVAL_LIMIT = 50
LLM_MODEL = "qwen3:1.7b"



def parse_args():
	parser = argparse.ArgumentParser(
		description="Evaluate ICD retrieval and LLM reranking."
	)

	parser.add_argument(
		"--semantic-mode",
		choices=["title", "enriched"],
		default="title"
	)

	# hybrid retrieval params
	parser.add_argument(
		"--retrieval-method",
		choices=["rrf", "weighted_rrf", "combsum"],
		default="rrf"
	)

	parser.add_argument(
		"--retrieval-lexical-weight",
		type=float,
		default=0.5
	)

	parser.add_argument(
		"--retrieval-semantic-weight",
		type=float,
		default=0.5
	)

	parser.add_argument(
		"--retrieval-rrf-k",
		type=int,
		default=60
	)

	# LLM fusion params
	parser.add_argument(
		"--fusion-retrieval-weight",
		type=float,
		default=0.3
	)

	parser.add_argument(
		"--fusion-llm-weight",
		type=float,
		default=0.7
	)

	parser.add_argument(
		"--fusion-rrf-k",
		type=int,
		default=60
	)

	parser.add_argument(
		"--output",
		type=str,
		default=None,
		help="Optional evaluation output filename."
	)

	return parser.parse_args()


def get_codes(results):
	return [result["code"] for result in results]


def save_candidates(results):
	fields = [
		"code",
		"title",
		"score",
		"parent_code",
		"exact_match",
		"lexical_rank",
		"semantic_rank",
		"rrf_score",
		"combined_score",
		"leaf"
	]

	return [
		{key: result[key] for key in fields if key in result}
		for result in results
	]


def get_gold_rank(codes, gold):
	for rank, code in enumerate(codes, start=1):
		if code in gold:
			return rank

	return None


def main():
	args = parse_args()

	if args.retrieval_rrf_k <= 0 or args.fusion_rrf_k <= 0:
		raise ValueError("Both retrieval and fusion RRF K must be positive")

	if args.retrieval_lexical_weight < 0 or args.retrieval_semantic_weight < 0:
		raise ValueError("Retrieval weights cannot be negative")

	if args.fusion_retrieval_weight < 0 or args.fusion_llm_weight < 0:
		raise ValueError("Fusion weights cannot be negative")

	if args.retrieval_method == "weighted_rrf":
		if args.retrieval_lexical_weight + args.retrieval_semantic_weight == 0:
			raise ValueError("At least one retrieval weight must be positive")

	if args.fusion_retrieval_weight + args.fusion_llm_weight == 0:
		raise ValueError("At least one fusion weight must be positive")


	OUTPUT_DIR.mkdir(exist_ok=True)

	if args.output:
		output_name = args.output
		if not output_name.endswith(".jsonl"):
			output_name += ".jsonl"
	else:
		output_name = (
			f"evaluation_{args.semantic_mode}_"
			f"{args.retrieval_method}_"
			f"retrieval-L{args.retrieval_lexical_weight:g}"
			f"-S{args.retrieval_semantic_weight:g}"
			f"-K{args.retrieval_rrf_k}_"
			f"fusion-R{args.fusion_retrieval_weight:g}"
			f"-LLM{args.fusion_llm_weight:g}"
			f"-K{args.fusion_rrf_k}.jsonl"
		)

	evaluation_file = OUTPUT_DIR / output_name

	with open(NOTES_FILE, "r", encoding="utf-8") as file:
		notes = [json.loads(line) for line in file if line.strip()]

	print("Building lexical index...")
	lexical_conn, parents = build_lexical_index(MMS_FILE)

	print("Building semantic index...")
	model, catalogue, embeddings, semantic_parents = build_semantic_index(
		MMS_FILE,
		semantic_mode = args.semantic_mode
	)

	catalogue_by_code = {
		entry["code"]: entry for entry in catalogue
	}

	print("Indexes are ready")
	print("Total notes:", len(notes))
	print("Saving all evaluation stages to:", evaluation_file)

	llm_errors = 0

	try:
		with open(evaluation_file, "w", encoding="utf-8") as output_file:
			for i, note in enumerate(notes, start=1):
				note_id = note["id"]
				note_text = note["text"]
				gold = note.get("gold", [])

				print(f"\nProcessing {i}/{len(notes)}: {note_id}")

				lexical_results = []
				semantic_results = []
				hybrid_candidates = []
				prepared_candidates = []

				llm_codes = []
				raw_response = ""
				raw_codes = []
				invalid_codes = []
				retrieval_error = ""
				llm_error = ""

				try:
					# Keep individual retrieval results for hit-rate analysis.
					lexical_results = lexical_search(
						note_text,
						lexical_conn,
						parents,
						limit=RETRIEVAL_LIMIT
					)

					semantic_results = semantic_search(
						note_text,
						model,
						catalogue,
						embeddings,
						semantic_parents,
						limit=RETRIEVAL_LIMIT
					)

					# Existing hybrid RRF retrieval.
					hybrid_candidates = hybrid_search(
						note_text,
						lexical_conn,
						parents,
						model,
						catalogue,
						embeddings,
						limit=RETRIEVAL_LIMIT,
						method=args.retrieval_method,
						K=args.retrieval_rrf_k,
						lexical_weight = args.retrieval_lexical_weight,
						semantic_weight = args.retrieval_semantic_weight
					)

					# Existing leaf-node and clinical keyword filtering.
					prepared_candidates = prepare_candidates(
						note_text,
						hybrid_candidates,
						catalogue_by_code
					)

				except Exception as e:
					retrieval_error = str(e)

				# Existing Qwen reranker and weighted RRF fusion.
				if prepared_candidates:
					try:
						result = rerank(
							note_text,
							prepared_candidates,
							model=LLM_MODEL
						)

						llm_codes = result["codes"]
						raw_response = result["raw_response"]
						raw_codes = result["raw_codes"]
						invalid_codes = result["invalid_codes"]

						if not llm_codes:
							llm_error = "LLM returned no valid codes"

					except Exception as e:
						llm_error = str(e)

					if llm_error:
						llm_errors += 1

				hybrid_codes = get_codes(prepared_candidates[:3])

				fallback_codes = list(llm_codes[:3])

				for code in hybrid_codes:
					if len(fallback_codes) == 3:
						break

					if code not in fallback_codes:
						fallback_codes.append(code)

				if llm_codes:
					fused_codes = fuse_candidates(
						prepared_candidates,
						llm_codes,
						retrieval_weight=args.fusion_retrieval_weight,
						llm_weight=args.fusion_llm_weight,
						k=args.fusion_rrf_k
					)
				else:
					fused_codes = hybrid_codes[:3]

				lexical_codes = get_codes(lexical_results)
				semantic_codes = get_codes(semantic_results)
				hybrid_all_codes = get_codes(hybrid_candidates)
				prepared_all_codes = get_codes(prepared_candidates)

				gold_set = set(gold)

				record = {
					"id": note_id,
					"style": note.get("style"),
					"text": note_text,
					"gold": gold,

					"retrieval": {
						"lexical": save_candidates(lexical_results),
						"semantic": save_candidates(semantic_results),
						"hybrid": save_candidates(hybrid_candidates),
						"prepared": save_candidates(prepared_candidates)
					},

					"predictions": {
						"lexical": lexical_codes[:3],
						"semantic": semantic_codes[:3],
						"hybrid": hybrid_codes,
						"prepared": prepared_all_codes[:3],
						"llm": llm_codes[:3],
						"fallback": fallback_codes,
						"fused": fused_codes
					},

					"gold_ranks": {
						"lexical": get_gold_rank(
							lexical_codes, gold_set
						),
						"semantic": get_gold_rank(
							semantic_codes, gold_set
						),
						"hybrid": get_gold_rank(
							hybrid_all_codes, gold_set
						),
						"prepared": get_gold_rank(
							prepared_all_codes, gold_set
						),
						"llm": get_gold_rank(
							llm_codes, gold_set
						),
						"fallback": get_gold_rank(
							fallback_codes, gold_set
						),
						"fused": get_gold_rank(
							fused_codes, gold_set
						)
					},

					"llm": {
						"model": LLM_MODEL,
						"raw_response": raw_response,
						"raw_codes": raw_codes,
						"invalid_codes": invalid_codes,
						"error": llm_error
					},

					"retrieval_error": retrieval_error
				}

				output_file.write(
					json.dumps(record, ensure_ascii=False) + "\n"
				)
				output_file.flush()

				print("Lexical:", lexical_codes[:3])
				print("Semantic:", semantic_codes[:3])
				print("Hybrid:  ", hybrid_codes)
				print("LLM:     ", llm_codes[:3])
				print("Fallback:", fallback_codes)
				print("Fused:   ", fused_codes)

				if retrieval_error:
					print("Retrieval error:", retrieval_error)

				if llm_error:
					print("LLM error:", llm_error)

	finally:
		lexical_conn.close()

	print("\nEvaluation run completed")
	print("Notes:", len(notes))
	print("LLM errors:", llm_errors)
	print("Saved:", evaluation_file)
	print("\nRun scripts/evaluate_fusion.py to compare methods and fusion weights.")


if __name__ == "__main__":
	main()