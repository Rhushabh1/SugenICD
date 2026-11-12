import os
import json
import time
# instead of http dependency
import urllib.request

from src.llm.config import get_model


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


# openai helper call
def call_openai(model, system_prompt, user_prompt, response_format):
	body = {
			"model": model,
			"input": [{
					"role": "system",
					"content": system_prompt
					},
					{
					"role": "user",
					"content": user_prompt
					}],
			"text": {
					"format": {
							"type": "json_schema",
							"name": "icd_codes",
							"strict": True,
							"schema": {
									"type": "object",
									"properties": {
												"codes": {
														"type": "array",
														"items": {
																"type": "string",
																"enum": response_format["properties"]["codes"]["items"]["enum"]
																},
														"maxItems": 3
														}
												},
									"required": ["codes"],
									"additionalProperties": False
									}
							}
					},
			"max_output_tokens": 100,
			"reasoning": {
						"effort": "none"
						}
			}

	request = urllib.request.Request("https://api.openai.com/v1/responses",
									data=json.dumps(body).encode("utf-8"),
									headers={
											"Content-Type": "application/json",
											"Authorization": "Bearer " + os.getenv("OPENAI_API_KEY")
											})

	start_time = time.time()

	with urllib.request.urlopen(request, timeout=120) as response:
		result = json.loads(response.read().decode("utf-8"))

	content = result.get("output_text", "")

	if not content:
		for item in result.get("output", []):
			for part in item.get("content", []):
				if part.get("type") == "output_text":
					content = part["text"]
					break

	# print("OpenAI time:", round(time.time() - start_time, 2), "sec")
	# print("OpenAI usage:", result.get("usage", {}))

	return content


# anthropic helper call
def call_anthropic(model, system_prompt, user_prompt, response_format):
	schema = {
			"type": "object",
			"properties": {
						"codes": {
								"type": "array",
								"items": {
										"type": "string",
										"enum": response_format["properties"]["codes"]["items"]["enum"]
										}
								}
						},
			"required": ["codes"],
			"additionalProperties": False
			}

	body = {
			"model": model,
			"max_tokens": 100,
			"system": system_prompt,
			"messages": [{
						"role": "user",
						"content": user_prompt
						}],
			"output_config": {
							"format": {
									"type": "json_schema",
									"schema": schema
									}
							}
			}

	request = urllib.request.Request("https://api.anthropic.com/v1/messages",
									data=json.dumps(body).encode("utf-8"),
									headers={
											"Content-Type": "application/json",
											"x-api-key": os.getenv("ANTHROPIC_API_KEY"),
											"anthropic-version": "2023-06-01"
											})

	start_time = time.time()

	with urllib.request.urlopen(request, timeout=120) as response:
		result = json.loads(response.read().decode("utf-8"))

	content = ""

	for block in result.get("content", []):
		if block.get("type") == "text":
			content = block["text"]
			break

	# print("Anthropic time:", round(time.time() - start_time, 2), "sec")
	# print("Anthropic usage:", result.get("usage", {}))

	return content


# migrated ollama helper call
def call_ollama(model, system_prompt, user_prompt, response_format):
	body = {
			"model": model,
			"messages": [{
						"role": "system",
						"content": system_prompt
						},
						{
						"role": "user",
						"content": user_prompt
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

	# API request for ollama
	url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/api/chat"
	request = urllib.request.Request(url,
									data = json.dumps(body).encode("utf-8"),
									headers = {"Content-Type": "application/json"})

	start_time = time.time()

	# send & receive response block
	with urllib.request.urlopen(request, timeout=120) as response:
		result = json.loads(response.read().decode("utf-8"))

	# parsing explicit json format
	content = result["message"]["content"].strip()

	# for LLM processing time diagnosis
	# print("Ollama time:", round(time.time() - start_time, 2), "sec")
	# print("Ollama load duration:", round(result.get("load_duration", 0) / 1e9, 2), "sec")
	# print("Ollama prompt tokens:", result.get("prompt_eval_count", 0))
	# print("Ollama generated tokens:", result.get("eval_count", 0))

	return content



# LLM reranking function (assumes candidates are already prepared)
def rerank(note, candidates, model = None, provider = "ollama"):
	if provider not in ["ollama", "openai", "anthropic"]:
		raise ValueError(f"Provider not implemented yet: {provider}")

	if model is None:
		model = get_model(provider)

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

	if provider == "ollama":
		content = call_ollama(model, SYSTEM_PROMPT, USER_PROMPT, response_format)

	elif provider == "openai":
		content = call_openai(model, SYSTEM_PROMPT, USER_PROMPT, response_format)

	elif provider == "anthropic":
		content = call_anthropic(model, SYSTEM_PROMPT, USER_PROMPT, response_format)

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
