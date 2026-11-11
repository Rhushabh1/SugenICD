import json
# instead of http dependency
import urllib.request


# single place where candidates are filtered before LLm sees them
# filter using hierarchy and clinical findings
def filter_candidates(note, candidates):
	note = note.lower()

	filtered_candidates = []

	for candidate in candidates:
		# exclude parent/non-leaf categories
		if not candidate.get("leaf", False):
			continue

		title = candidate["title"].lower()

		# exclude complication-related codes when note explicitly says uncomplicated
		if "uncomplicated" in note:
			title_words = title.split()

			if "severe" in title_words or "complicated" in title_words:
				continue

			# to not get confused with "without complications"
			if "with" in title_words and ("complication" in title_words or "complications" in title_words):
				continue

		filtered_candidates.append(candidate)

	return filtered_candidates


# LLM reranking function
def rerank(note, candidates, catalogue_by_code, model = "qwen3:1.7b"):
	# add leaf metadata
	for candidate in candidates:
		entry = catalogue_by_code.get(candidate["code"])
		if entry:
			candidate["leaf"] = entry["leaf"]

	# basic leaf node and keyword filtering
	candidates = filter_candidates(note, candidates)
	candidate_list = []

	# # extract all the relevant LLM info
	for candidate in candidates:
		candidate_list.append({
							"code": candidate["code"],
							"title": candidate["title"],
							# "parent_code": candidate.get("parent_code"),
							# "leaf": candidate.get("leaf"),
							# LLM misinterpreting residual flag as a clinical finding
							# "residual": candidate.get("residual"),
							# "depth": candidate.get("depth")
							})

	# prompt construction
	PROMPT = f"""
You are an ICD-11 clinical coding assistant.
Select the most appropriate ICD-11 diagnosis codes for the clinical note.

CLINICAL NOTE:
{note}

AVAILABLE ICD-11 CANDIDATE CODES:
{json.dumps(candidate_list, ensure_ascii = False)}

Instructions:
- Select codes only from the supplied candidates
- Only select terminal diagnosis categories (leaf categories)
- Choose the most specific diagnosis supported by the clinical note
- Do not select a code if its description contradicts the clinical note
- Do not infer complications, severity or other clinical details that are not documented
- If a more specific child requires details that are absent from the note, select the appropriate unspecified (.Z) category when available
- Do not confuse unspecified categories with other residual categories. Read their complete descriptions carefully
- Return at most 3 distinct codes, ordered from most to least appropriate
- If no candidate is supported, return an empty list
- Do not invent or modify ICD-11 codes

Before selecting codes, check each candidate against the clinical note:
- Is its diagnosis supported?
- Does its description contradict any documented finding?
- Does it require clinical details that are absent?
- Is there a more specific supported candidate?

Return only valid JSON in this format:
{{
	"codes": ["CODE1", "CODE2"],
	"reason": "Brief explanation"
}}
"""

	body = {
			"model": model,
			"messages": [{
						"role": "user",
						"content": PROMPT
						}],
			"format": "json",
			"think": False,
			"stream": False,
			"options": {
						# no randomness -> need higher recall
						"temperature": 0,
						"num_predict": 200
						}
			}

	# API request for ollama
	request = urllib.request.Request("http://localhost:11434/api/chat",
									data = json.dumps(body).encode("utf-8"),
									headers = {"Content-Type": "application/json"})

	# send & receive response block
	with urllib.request.urlopen(request, timeout = 180) as response:
		result = json.loads(response.read().decode("utf-8"))

	# parsing explicit json format
	content = result["message"]["content"].strip()

	if not content:
		print("Ollama returned empty response")
		return {
				"codes": [],
				"reason": "",
				# for error analysis
				"raw_response": "",
				# codes originally returned
				"raw_codes": [],
				# returned codes which were invalid
				"invalid_codes": [],
				# candidates actually sent to LLM
				"candidate_codes": [candidate["code"] for candidate in candidates],
				"candidate_details": [{
									"code": candidate["code"], 
									"title": candidate["title"]
									} for candidate in candidates]
				}

	output = json.loads(content)

	# validate codes against retrieved candidates
	valid_codes = {candidate["code"] for candidate in candidates if candidate["leaf"]}

	# extract top 3 codes from output
	selected_codes = []
	invalid_codes = []
	raw_codes = output.get("codes", [])
	
	for code in raw_codes:
		if code in valid_codes and code not in selected_codes:
			selected_codes.append(code)
		elif code not in valid_codes:
			invalid_codes.append(code)

		if len(selected_codes) == 3:
			break

	return {
			"codes": selected_codes,
			"reason": output.get("reason", ""),
			"raw_response": content,
			"raw_codes": raw_codes,
			"invalid_codes": invalid_codes,
			"candidate_codes": [candidate["code"] for candidate in candidates],
			"candidate_details": [{
								"code": candidate["code"], 
								"title": candidate["title"]
								} for candidate in candidates]
			}
