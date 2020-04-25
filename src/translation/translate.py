import six


def translate_text(target, wordSet):

	transFile = open('trans.txt', 'w')
	# [START translate_translate_text]
	"""Translates text into the target language.
	Target must be an ISO 639-1 language code.
	See https://g.co/cloud/translate/v2/translate-reference#supported_languages
	"""
	from google.cloud import translate_v2 as translate
	translate_client = translate.Client()

	# if isinstance(text, six.binary_type):
	#     text = text.decode('utf-8')

	# Text can also be a sequence of strings, in which case this method
	# will return a sequence of results for each text.

	querySize = 100

	for i in range(0, len(wordSet), querySize):
		text = wordSet[i:min(len(wordSet)-1, i+querySize)]
		print(text)
		results = translate_client.translate(
			text, target_language=target)

		for result in results:
			transFile.write(result['input'] + ' ' + result['translatedText'] + '\n')
			# print(u'Text: {}'.format(result['input']))
			# print(u'Translation: {}'.format(result['translatedText']))
			# print(u'Detected source language: {}'.format(
			# 	result['detectedSourceLanguage']))

	transFile.close()
	# [END translate_translate_text]

wordFile = open('wordFile.txt', 'r')

transFile = open('trans.txt', 'w')

wordSet = []

for word in wordFile:
	word = word[:-1]
	wordSet.append(word)


translate_text('en', wordSet)
