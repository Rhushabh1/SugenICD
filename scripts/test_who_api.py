import base64
import json
import os
import urllib.parse
import urllib.request


TOKEN_URL = "https://icdaccessmanagement.who.int/connect/token"
API_VERSION = "v2"


def get_token():
	client_id = os.environ.get("WHO_CLIENT_ID")
	client_secret = os.environ.get("WHO_CLIENT_SECRET")

	if not client_id or not client_secret:
		raise ValueError("WHO_CLIENT_ID or WHO_CLIENT_SECRET is missing")

	credentials = f"{client_id}:{client_secret}".encode("utf-8")
	encoded_credentials = base64.b64encode(credentials).decode("utf-8")

	data = urllib.parse.urlencode({
		"grant_type": "client_credentials",
		"scope": "icdapi_access"
	}).encode("utf-8")

	request = urllib.request.Request(
		TOKEN_URL,
		data=data,
		headers={
			"Authorization": f"Basic {encoded_credentials}",
			"Content-Type": "application/x-www-form-urlencoded"
		}
	)

	with urllib.request.urlopen(request, timeout=30) as response:
		result = json.loads(response.read().decode("utf-8"))

	return result["access_token"]


def main():
	token = get_token()

	# This is the URI already present in the assignment catalogue.
	category_uri = (
		"https://id.who.int/icd/release/11/2026-01/mms/257068234"
	)

	request = urllib.request.Request(
		category_uri,
		headers={
			"Authorization": f"Bearer {token}",
			"Accept": "application/json",
			"Accept-Language": "en",
			"API-Version": API_VERSION
		}
	)

	with urllib.request.urlopen(request, timeout=30) as response:
		category = json.loads(response.read().decode("utf-8"))

	print("WHO API connection successful")
	print("Category:", category.get("title"))
	print("\nAvailable fields:")
	print(", ".join(sorted(category.keys())))

	fields = [
		"synonym",
		"inclusion",
		"indexTerm",
		"exclusion",
		"definition",
		"longDefinition",
		"codingNote",
		"note"
	]

	print("\nTerminology fields:")
	for field in fields:
		if field in category:
			print(f"\n{field}:")
			print(json.dumps(category[field], indent=2, ensure_ascii=False)[:1500])


if __name__ == "__main__":
	main()