# for RRF of retrieval and LLM after LLM reranker returns codes
def fuse_candidates(candidates, llm_codes, retrieval_weight = 0.3, llm_weight = 0.7, k = 60):
	llm_ranks = {
		code: rank
		for rank, code in enumerate(llm_codes, start=1)
	}

	scores = {}
	
	# Retrieval contribution
	for rank, candidate in enumerate(candidates, start=1):
		code = candidate["code"]
		scores[code] = scores.get(code, 0.0) + (
			retrieval_weight / (k + rank)
		)

	# LLM contribution
	for rank, code in enumerate(llm_codes, start=1):
		scores[code] = scores.get(code, 0.0) + (
			llm_weight / (k + rank)
		)

	ranked = [
		{"code": code, "score": score}
		for code, score in scores.items()
		if score > 0
	]

	ranked.sort(key=lambda x: x["score"], reverse=True)

	return [item["code"] for item in ranked[:3]]
