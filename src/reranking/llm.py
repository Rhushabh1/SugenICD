import json
# instead of http dependency
import urllib.request


# LLM reranking function
def rerank(note, candidates, model = "qwen3:1.7b"):
	candidate_list = []

	# extract all the relevant LLM info
	for candidate in candidates:
		candidate_list.append({
							"code": candidate["code"],
							"title": candidate["title"]
							})

	# prompt construction
	PROMPT = f"""
You are an ICD-11 clinical coding assistant.
Select the most appropriate ICD-11 diagnosis codes for the clinical note.

CLINICAL NOTE:
{note}

AVAILABLE ICD-11 CANDIDATES:
{json.dumps(candidate_list, ensure_ascii = False)}

Instructions:
- Select codes only from the available candidates
- Choose the most specific diagnosis supported by the clinical note
- Do not assume symptoms, complications or details that are not documented
- Prefer a specific diagnosis over a general parent category when supported
- Do not select unrelated symptom codes when a supported diagnosis is available
- Return at most 3 codes, ordered from most to least appropriate
- If no candidate is supported, return an empty list
- Do not invent or modify ICD-11 codes

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
						"num_predict": 300
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
		print(result)
		return {
				"codes": [],
				"reason": ""
				}

	output = json.loads(content)

	# validate codes against retrieved candidates
	valid_codes = {candidate["code"] for candidate in candidates}

	# extract top 3 codes from output
	selected_codes = []
	for code in output.get("codes", []):
		if code in valid_codes and code not in selected_codes:
			selected_codes.append(code)

		if len(selected_codes) == 3:
			break

	return {
			"codes": selected_codes,
			"reason": output.get("reason", "")
			}