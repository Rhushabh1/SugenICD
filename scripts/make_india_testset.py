
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MMS_FILE = ROOT / "data" / "mms-2026-01.json"
OUTPUT_FILE = ROOT / "data" / "india_generalization.jsonl"


# synthetic notes. Each group describes the same condition in poor English, Hindi and Marathi respectively
# limited to just 6 conditions -> type 2 diabetes, essential hypertension, migraine, gastro-oesophageal reflux, iron-deficiency anaemia and knee osteoarthritis
CASES = [
	{
		"key": "type2_diabetes",
		"query": "type 2 diabetes mellitus",
		"prefer": "without complications",
		"notes": [
			(
				"54F tired for months, thirsty all time, passing urine "
				"many times esp night. Fasting glucose 168, HbA1c 8.1. "
				"T2DM, no documented complications. Review sugars."
			),
			(
				"54 वर्ष महिला, कई महीनों से थकान और बहुत प्यास लगती है। "
				"बार बार पेशाब आता है, रात में भी। शुगर जांच बढ़ी हुई, "
				"HbA1c 8.1। डॉक्टर ने टाइप 2 मधुमेह लिखा, "
				"अभी कोई complication दर्ज नहीं।"
			),
			(
				"54 वर्ष महिला, खूप तहान लागते आणि वारंवार लघवी होते, "
				"रात्रीसुद्धा. HbA1c 8.1. तपासणीत साखर वाढलेली. "
				"टाइप 2 मधुमेह, सध्या complications नोंदलेले नाहीत."
			),
		],
	},
	{
		"key": "hypertension",
		"query": "essential hypertension",
		"prefer": "",
		"notes": [
			(
				"49M headache on/off, BP 168/96 today and similar at "
				"last 2 visits. No acute end organ symptoms. "
				"Assessment: essential hypertension."
			),
			(
				"49 साल पुरुष, कभी कभी सिर दर्द रहता है। आज बीपी 168/96, "
				"पिछली जांच में भी बढ़ा था। कोई सीने में दर्द या सांस की "
				"तकलीफ नहीं। जांच में उच्च रक्तचाप।"
			),
			(
				"49 वर्ष पुरुष, अधूनमधून डोके दुखते. आज रक्तदाब 168/96, "
				"मागील भेटीतही जास्त होता. छातीत दुखणे किंवा श्वासाचा "
				"त्रास नाही. उच्च रक्तदाब."
			),
		],
	},
	{
		"key": "migraine",
		"query": "migraine",
		"prefer": "without aura",
		"notes": [
			(
				"28F recurrent one-sided throbbing headache, nausea, "
				"light bothers her, lasts 6-8 hrs. No visual warning "
				"or aura. Assessment: migraine without aura."
			),
			(
				"28 साल महिला, सिर के एक तरफ तेज धड़कने जैसा दर्द, "
				"उल्टी जैसा लगता है और रोशनी से परेशानी। 6-8 घंटे रहता है। "
				"आंखों के आगे चमक या aura नहीं। माइग्रेन।"
			),
			(
				"28 वर्ष महिला, डोक्याच्या एका बाजूला ठणकणारी वेदना, "
				"मळमळ आणि प्रकाशाचा त्रास. 6-8 तास टिकते. "
				"दृष्टीसमोर चमक किंवा aura नाही. अर्धशिशी."
			),
		],
	},
	{
		"key": "gerd",
		"query": "gastro-oesophageal reflux",
		"prefer": "without oesophagitis",
		"notes": [
			(
				"42M burning behind chest after meals, sour fluid comes "
				"up when lying down. Better with antacid. No dysphagia "
				"or bleeding. GERD, no documented oesophagitis."
			),
			(
				"42 साल पुरुष, खाना खाने के बाद सीने में जलन होती है, "
				"लेटने पर खट्टी डकार और पानी मुंह में आता है। "
				"antacid से आराम। निगलने में परेशानी नहीं। "
				"गैस्ट्रो इसोफेजियल रिफ्लक्स।"
			),
			(
				"42 वर्ष पुरुष, जेवणानंतर छातीत जळजळ होते, "
				"झोपल्यावर आंबट ढेकर येतात. antacid ने आराम. "
				"गिळताना त्रास नाही. gastro-oesophageal reflux."
			),
		],
	},
	{
		"key": "iron_deficiency",
		"query": "iron deficiency anaemia",
		"prefer": "unspecified",
		"notes": [
			(
				"31F fatigue, dizzy on standing, pale conjunctiva. "
				"Hb 8.9 g/dL, MCV low, ferritin low. Assessment: "
				"iron deficiency anaemia."
			),
			(
				"31 साल महिला, बहुत थकान, खड़े होने पर चक्कर, "
				"चेहरा पीला। Hb 8.9, MCV कम और ferritin कम। "
				"खून की कमी, लोह की कमी से होने वाला एनीमिया।"
			),
			(
				"31 वर्ष महिला, खूप थकवा, उभे राहिल्यावर चक्कर, "
				"चेहरा फिकट. Hb 8.9, MCV आणि ferritin कमी. "
				"रक्ताची कमतरता, लोहाची कमतरता."
			),
		],
	},
	{
		"key": "knee_osteoarthritis",
		"query": "osteoarthritis",
		"prefer": "knee",
		"notes": [
			(
				"63F knee pain for 2 years, worse on walking and stairs, "
				"morning stiffness brief. Crepitus, X-ray joint-space "
				"narrowing and osteophytes. Knee osteoarthritis."
			),
			(
				"63 साल महिला, दो साल से घुटनों में दर्द, चलने और "
				"सीढ़ी चढ़ने पर ज्यादा। सुबह थोड़ी अकड़न। जांच में "
				"घुटनों के जोड़ में घिसाव और osteophytes।"
			),
			(
				"63 वर्ष महिला, दोन वर्षांपासून गुडघेदुखी, चालताना "
				"आणि जिना चढताना वाढते. सकाळी थोडा कडकपणा. "
				"एक्स-रे मध्ये सांध्यांची झीज आणि osteophytes."
			),
		],
	},
]


def load_catalogue():
	with open(MMS_FILE, "r", encoding="utf-8") as file:
		return json.load(file)


def resolve_leaf_code(catalogue, query, prefer=""):
	"""
	Resolve the synthetic diagnosis to a real leaf code in the
	supplied official MMS catalogue. Never invent a code.
	"""
	query = query.lower()
	prefer = prefer.lower()

	matches = [
		entry for entry in catalogue
		if entry.get("leaf") is True
		and query in entry.get("title", "").lower()
	]

	if not matches:
		raise ValueError(
			f"No leaf MMS category found for: {query}. "
			"Check the official catalogue title."
		)

	if prefer:
		preferred = [
			entry for entry in matches
			if prefer in entry.get("title", "").lower()
		]
		if preferred:
			matches = preferred

	# Prefer the shortest matching leaf title.
	matches.sort(key=lambda entry: (
		len(entry.get("title", "")),
		entry["code"]
	))

	return matches[0]


def main():
	catalogue = load_catalogue()
	output = []

	print("Resolved official MMS leaf codes:")
	print("---------------------------------")

	for case in CASES:
		entry = resolve_leaf_code(
			catalogue,
			case["query"],
			case["prefer"]
		)

		code = entry["code"]
		title = entry["title"]

		print(f'{case["key"]}: {code} | {title}')

		styles = ["poor_english", "hindi", "marathi"]

		for index, text in enumerate(case["notes"]):
			output.append({
				"id": f'india-{case["key"]}-{index + 1:02d}',
				"text": text,
				"gold": [code],
				"style": styles[index],
			})

	OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

	with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
		for note in output:
			file.write(
				json.dumps(note, ensure_ascii=False) + "\n"
			)

	print("---------------------------------")
	print(f"Generated {len(output)} synthetic notes.")
	print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
	main()
