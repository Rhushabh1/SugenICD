import json
import time
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


# prepare candidates with metadata and filtering before LLM reranking
def prepare_candidates(note, candidates, catalogue_by_code):
	# add leaf metadata from ICD catalogue
	for candidate in candidates:
		entry = catalogue_by_code.get(candidate["code"])

		if entry:
			candidate["leaf"] = entry["leaf"]

	# apply common candidate filters
	return filter_candidates(note, candidates)


# LLM reranking function (assumes candidates are already prepared)
def rerank(note, candidates, model = "qwen3:1.7b"):
	candidate_list = []

	# # extract all the relevant LLM info
	for candidate in candidates:
		candidate_list.append(candidate["code"] + ": " + candidate["title"])
							# "parent_code": candidate.get("parent_code"),
							# "leaf": candidate.get("leaf"),
							# LLM misinterpreting residual flag as a clinical finding
							# "residual": candidate.get("residual"),
							# "depth": candidate.get("depth")
							# })

	candidate_codes = [candidate["code"] for candidate in candidates]

	if not candidate_codes:
		return {
				"codes": [],
				# "reason": "",
				"raw_response": "",
				"raw_codes": [],
				"invalid_codes": [],
				"candidate_codes": [],
				"candidate_details": []
				}

	# constrained candidate codes -> to prevent hallucinations, duplicates, malformed jsons
	response_format = {
					"type": "object",
					"properties": {
								"codes": {
										"type": "array",
										"items": {
												"type": "string",
												"enum": candidate_codes
												},
										"maxItems": 3,
										"uniqueItems": True
										},
								# "reason": {
								# 		"type": "string"
								# 		}
								},
					"required": ["codes"],
					"additionalProperties": False
					}

	# prompt construction
	SYSTEM_PROMPT = """
You are an ICD-11 clinical coding assistant.
Select the most appropriate ICD-11 diagnosis codes for the clinical note.

RULES:
- Select codes only from the supplied candidates.
- Select only terminal diagnosis categories.
- Choose the most specific diagnosis supported by the note.
- Do not infer severity, complications, causes, organisms or subtypes.
- If specific details are absent, prefer an appropriate unspecified category.
- Do not select codes that contradict the note.
- Return at most 3 distinct codes, ordered by relevance.
- Return an empty list if no candidate is supported.
- Never invent or modify ICD-11 codes.

Return only valid JSON in this format:
{{
	"codes": ["CODE1", "CODE2"],
}}
"""
	
	USER_PROMPT = f"""
CLINICAL NOTE:
{note}

AVAILABLE ICD-11 CANDIDATE CODES:
{json.dumps(candidate_list, ensure_ascii = False, separators = (",", ":"))}

Select the best supported codes from these candidates.
"""

	body = {
			"model": model,
			"messages": [{
						"role": "system",
						"content": SYSTEM_PROMPT
						},
						{
						"role": "user",
						"content": USER_PROMPT
						}],
			# "format": "json",
			"format": response_format,
			"think": False,
			"stream": False,
			# keep model loaded between notes instead of unloading
			"keep_alive": "30m",
			"options": {
						# no randomness -> need higher recall
						"temperature": 0,
						# number of tokens generated (codes + short reason)
						"num_predict": 40,
						# limits context allocation
						"num_ctx": 2048
						}
			}

	start_time = time.time()

	# API request for ollama
	request = urllib.request.Request("http://localhost:11434/api/chat",
									data = json.dumps(body).encode("utf-8"),
									headers = {"Content-Type": "application/json"})

	# send & receive response block
	with urllib.request.urlopen(request, timeout = 180) as response:
		result = json.loads(response.read().decode("utf-8"))

	# for LLM processing time diagnosis
	print("LLM time:", round(time.time() - start_time, 2), "sec")
	print("Model load:", round(result["load_duration"] / 1e9, 2), "sec")
	print("Prompt processing:", round(result["prompt_eval_duration"] / 1e9, 2), "sec")
	print("Answer generation:", round(result["eval_duration"] / 1e9, 2), "sec")
	print("Prompt tokens:", result["prompt_eval_count"])
	print("Generated tokens:", result["eval_count"])

	# parsing explicit json format
	content = result["message"]["content"].strip()

	if not content:
		print("Ollama returned empty response")
		return {
				"codes": [],
				# "reason": "",
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
	valid_codes = {candidate["code"] for candidate in candidates}

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
			# "reason": output.get("reason", ""),
			"raw_response": content,
			"raw_codes": raw_codes,
			"invalid_codes": invalid_codes,
			"candidate_codes": [candidate["code"] for candidate in candidates],
			"candidate_details": [{
								"code": candidate["code"], 
								"title": candidate["title"]
								} for candidate in candidates]
			}
