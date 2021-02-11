import stanza
from nltk import word_tokenize
import re

class Truecase():

	def __init__(self):
		self.nlp = stanza.Pipeline('de', processors='pos,ner,tokenize,mwt', batch_size=3000, logging_level="WARN")


	def _parse(self, text):
		parsed = []
		doc = self.nlp(text)
		for s in doc.sentences:
			sent = []
			for t in s.tokens:
				for w in t.words:
					sent.append((w.text, w.upos, t.ner))
			parsed.append(sent)
		return parsed

	def _assemble(self, parsed_text):
		result = []
		for s in parsed_text:
			sent = []
			for tup in s:
				#todo: capitalize if w at sent start after quotes
				#todo: check why articles are capitalized
				if s.index(tup) == 0 or tup[1] == 'NOUN' or tup[1] == 'PROPN' or tup[2] != "O":
					sent.append(tup[0].capitalize())
				else:
					sent.append(tup[0])
			result.extend(sent)
		result = " ".join(result)
		result = re.sub(r" (?=:|;|'|\.|,|[!]|[?])", "", result)
		#(?:(?<=a)|(?<=bc)))
		#todo: parse replace "am", "zur"
		result = re.sub(r"(?:(?<=bei)|(?<=vo|zu)|(?<=i))n? dem","m", result)
		return result


	def truecase(self, text):
		text = text.lower()
		parsed_text = self._parse(text)
		truecased = self._assemble(parsed_text)
		return truecased

	def accuracy(self, test_data):
		"""test_data is a string with correct casing"""
		#todo: accuracy should run on big chunks of data, check if compared tokens are the same
		tp = 0
		truecased = self.truecase(test_data)
		result = word_tokenize(truecased, "german")
		test_data = word_tokenize(test_data, "german")
		for tok1, tok2 in zip(result, test_data):
			if tok1 == tok2:
				tp += 1
		return tp / len(result)

