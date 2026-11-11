import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock


TOKEN_URL = "https://icdaccessmanagement.who.int/connect/token"
API_URL = "https://id.who.int"

MMS_FILE = Path("data/mms-2026-01.json")
PROGRESS_FILE = Path("data/who_enrichment_progress.jsonl")
OUTPUT_FILE = Path("data/mms-2026-01-enriched.json")
ERROR_FILE = Path("data/who_enrichment_errors.jsonl")

WORKERS = 4

token = None
token_expiry = 0
token_lock = Lock()


def get_token(force=False):
	global token, token_expiry

	with token_lock:
		if token and time.time() < token_expiry - 60 and not force:
			return token

		client_id = os.environ.get("WHO_CLIENT_ID")
		client_secret = os.environ.get("WHO_CLIENT_SECRET")

		if not client_id or not client_secret:
			raise ValueError("WHO credentials are missing")

		credentials = f"{client_id}:{client_secret}".encode("utf-8")
		encoded = base64.b64encode(credentials).decode("utf-8")

		data = urllib.parse.urlencode({
			"grant_type": "client_credentials",
			"scope": "icdapi_access"
		}).encode("utf-8")

		request = urllib.request.Request(
			TOKEN_URL,
			data=data,
			headers={
				"Authorization": f"Basic {encoded}",
				"Content-Type": "application/x-www-form-urlencoded"
			}
		)

		with urllib.request.urlopen(request, timeout=30) as response:
			result = json.loads(response.read().decode("utf-8"))

		token = result["access_token"]
		token_expiry = time.time() + result.get("expires_in", 3600)

		return token


def get_text(value):
	if isinstance(value, str):
		return [value.strip()] if value.strip() else []

	if isinstance(value, list):
		results = []
		for item in value:
			results.extend(get_text(item))
		return results

	if isinstance(value, dict):
		# WHO terminology records can contain nested labels.
		for key in ["label", "@value", "value", "title", "term"]:
			if key in value:
				return get_text(value[key])

	return []


def get_field(category, field):
	values = get_text(category.get(field, []))

	# Remove duplicates while preserving their original order.
	return list(dict.fromkeys(values))


def make_url(entry):
	# Use the release-specific MMS URI from the 2026-01 catalogue.
	uri = entry["uri"].replace(
		"/release/11/mms/",
		"/release/11/2026-01/mms/"
	)

	return uri.replace("http://", "https://", 1)


def fetch_category(entry):
	url = make_url(entry)

	for attempt in range(5):
		try:
			request = urllib.request.Request(
				url,
				headers={
					"Authorization": f"Bearer {get_token()}",
					"Accept": "application/json",
					"Accept-Language": "en",
					"API-Version": "v2"
				}
			)

			with urllib.request.urlopen(request, timeout=45) as response:
				category = json.loads(response.read().decode("utf-8"))

			return {
				"code": entry["code"],
				"fully_specified_name": get_field(category, "fullySpecifiedName"),
				"index_terms": get_field(category, "indexTerm"),
				"inclusion_terms": get_field(category, "inclusion"),
				"exclusions": get_field(category, "exclusion"),
				"definitions": get_field(category, "definition"),
				"coding_notes": (
					get_field(category, "codingNote")
					+ get_field(category, "note")
				)
			}

		except urllib.error.HTTPError as error:
			if error.code == 401:
				get_token(force=True)
			elif error.code == 429 or error.code >= 500:
				time.sleep(2 ** attempt)
			else:
				return {
					"code": entry["code"],
					"error": f"HTTP {error.code}: {error.reason}"
				}

		except Exception as error:
			if attempt == 4:
				return {
					"code": entry["code"],
					"error": str(error)
				}

			time.sleep(2 ** attempt)

	return {
		"code": entry["code"],
		"error": "Request failed after retries"
	}


def load_progress():
	completed = {}

	if not PROGRESS_FILE.exists():
		return completed

	with open(PROGRESS_FILE, "r", encoding="utf-8") as file:
		for line in file:
			if line.strip():
				result = json.loads(line)
				completed[result["code"]] = result

	return completed


def main():
	with open(MMS_FILE, "r", encoding="utf-8") as file:
		catalogue = json.load(file)

	completed = load_progress()

	remaining = [
		entry for entry in catalogue
		if entry["code"] not in completed
	]

	print("Total catalogue entries:", len(catalogue))
	print("Already downloaded:", len(completed))
	print("Remaining:", len(remaining))

	get_token()

	errors = []

	with open(PROGRESS_FILE, "a", encoding="utf-8") as progress_file:
		with ThreadPoolExecutor(max_workers=WORKERS) as executor:
			futures = {
				executor.submit(fetch_category, entry): entry
				for entry in remaining
			}

			count = len(completed)

			for future in as_completed(futures):
				result = future.result()
				count += 1

				if "error" in result:
					errors.append(result)
				else:
					completed[result["code"]] = result

					progress_file.write(
						json.dumps(result, ensure_ascii=False) + "\n"
					)
					progress_file.flush()

				if count % 100 == 0 or count == len(catalogue):
					print(
						f"Processed {count}/{len(catalogue)} "
						f"| Successful: {len(completed)} "
						f"| Errors: {len(errors)}"
					)

	# Save errors separately so failed records can be retried.
	if errors:
		with open(ERROR_FILE, "w", encoding="utf-8") as file:
			for error in errors:
				file.write(json.dumps(error) + "\n")

	# Merge WHO terminology into the original MMS catalogue.
	enriched_catalogue = []

	for entry in catalogue:
		result = completed.get(entry["code"], {})

		enriched_entry = entry.copy()

		enriched_entry["fully_specified_name"] = result.get("fully_specified_name", [])
		enriched_entry["index_terms"] = result.get("index_terms", [])
		enriched_entry["inclusion_terms"] = result.get("inclusion_terms", [])
		enriched_entry["exclusions"] = result.get("exclusions", [])
		enriched_entry["definitions"] = result.get("definitions", [])
		enriched_entry["coding_notes"] = result.get("coding_notes", [])

		# enriched_entry["synonyms"] = result.get("synonyms", [])
		
		enriched_catalogue.append(enriched_entry)

	with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
		json.dump(enriched_catalogue, file, ensure_ascii=False)

	print("\nDownload finished")
	print("Catalogue entries:", len(enriched_catalogue))
	print("Successfully enriched:", len(completed))
	print("Errors:", len(errors))
	print("Output:", OUTPUT_FILE)

	if errors:
		print("Failed records:", ERROR_FILE)
		print("Run the script again to retry failed records.")


if __name__ == "__main__":
	main()