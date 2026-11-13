import argostranslate.package

argostranslate.package.update_package_index()
packages = argostranslate.package.get_available_packages()

matches = [
	p for p in packages
	if p.from_code == "hi" and p.to_code == "en"
]

for p in matches:
	print(p)

import argostranslate.package
import argostranslate.translate

package = matches[0]
argostranslate.package.install_from_path(package.download())

print(
	argostranslate.translate.translate(
		"मुझे पिछले तीन दिनों से तेज बुखार है और खांसी नहीं है।",
		"hi",
		"en"
	)
)