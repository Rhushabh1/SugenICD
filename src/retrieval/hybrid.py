from src.retrieval.lexical import search as lexical_search
from src.retrieval.semantic import search as semantic_search


# benchmarking RRF, weighted RRF, combSUM rerankers
def normalize_scores(results, higher_is_better = True):
	if not results:
		return {}

	scores = [result["score"] for result in results]
	minimum = min(scores)
	maximum = max(scores)

	normalized = {}

	for result in results:
		code = result["code"]

		if maximum == minimum:
			score = 1.0
		elif higher_is_better:
			score = (result["score"] - minimum) / (maximum - minimum)
		else:
			score = (maximum - result["score"]) / (maximum - minimum)

		normalized[code] = score

	return normalized




# combine lexical and semantic rankings using RRF (K = 60)
# same parents mapping for both retrievers (constant across retrievers)
def search(note, lexical_conn, parents, model, catalogue, embeddings, limit = 50, method = "rrf", K = 60):
	# gather both results
	lexical_results = lexical_search(note, lexical_conn, parents, limit = limit)
	semantic_results = semantic_search(note, model, catalogue, embeddings, parents, limit = limit)

	# normalize scores for rerankers
	lexical_scores = normalize_scores(lexical_results, higher_is_better = False)
	semantic_scores = normalize_scores(semantic_results, higher_is_better = True)

	# no duplicate candidates in combined
	combined = {}

	# add lexical rankings
	for rank, result in enumerate(lexical_results, start = 1):
		code = result["code"]

		# if code not in combined:
		combined[code] = {
						"code": code,
						"title": result["title"],
						"parent_code": result["parent_code"],
						"lexical_rank": rank,
						"semantic_rank": None,
						"rrf_score": 0,
						"combined_score": 0
						}

		# # assuming code is in combined
		# combined[code]["lexical_rank"] = rank 
		# combined[code]["rrf_score"] += 1 / (K + rank)

	# add semantic rankings
	for rank, result in enumerate(semantic_results, start = 1):
		code = result["code"]

		if code not in combined:
			combined[code] = {
							"code": code,
							"title": result["title"],
							"parent_code": result["parent_code"],
							"lexical_rank": None,
							"semantic_rank": None,
							"rrf_score": 0,
							"combined_score": 0
							}

		# assuming code is in combined
		combined[code]["semantic_rank"] = rank 
		# combined[code]["rrf_score"] += 1 / (K + rank)

	# calculate fusion score
	for code, result in combined.items():
		lexical_rank = result["lexical_rank"]
		semantic_rank = result["semantic_rank"]

		# equal importance to both rankings
		if method == "rrf":
			if lexical_rank is not None:
				result["rrf_score"] += 1 / (K + lexical_rank)
			if semantic_rank is not None:
				result["rrf_score"] += 1 / (K + semantic_rank)

			result["combined_score"] = result["rrf_score"]

		# lexical = 0.7, semantic = 0.3
		elif method == "weighted_rrf":
			if lexical_rank is not None:
				result["combined_score"] += 0.7 / (K + lexical_rank)
			if semantic_rank is not None:
				result["combined_score"] += 0.3 / (K + semantic_rank)

		# normalize scores to 0-1 and then combine then using same weighted RRF
		elif method == "combsum":
			result["combined_score"] = 0.7 * lexical_scores.get(code, 0) + 0.3 * semantic_scores.get(code, 0)


	results = list(combined.values())
	# final sorting by RRF
	results.sort(key = lambda x: x["combined_score"], reverse = True)

	return results[:limit]

