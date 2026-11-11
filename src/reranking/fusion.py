# for RRF of retrieval and LLM after LLM reranker returns codes
def fuse_candidates(candidates, llm_codes, retrieval_weight = 0.3, llm_weight = 0.7, k = 60):
    llm_ranks = {}

    # store LLM ranks for quick mapping
    for rank, code in enumerate(llm_codes, start=1):
        llm_ranks[code] = rank

    scores = []

    # store score for candidates
    for rank, candidate in enumerate(candidates, start=1):
        code = candidate["code"]

        # weighted RRF formula = w / (k + rank)
        score = retrieval_weight / (k + rank)

        if code in llm_ranks:
            score += llm_weight / (k + llm_ranks[code])

        scores.append({
            "code": code,
            "score": score
        })

    scores.sort(key=lambda x: x["score"], reverse=True)

    return [item["code"] for item in scores[:3]]
