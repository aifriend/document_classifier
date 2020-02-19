import spacy

from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher
from spacy.pipeline import EntityRuler

nlp = spacy.load("es_core_news_md")


class NlpMatcher:

    @staticmethod
    def token_matcher(pattern, text):
        matcher = Matcher(nlp.vocab)
        matcher.add("Matching", None, pattern)

        doc = nlp(text)
        matches = matcher(doc)
        for match_id, start, end in matches:
            # Get the string representation
            string_id = nlp.vocab.strings[match_id]
            span = doc[start:end]  # The matched span
            print(match_id, string_id, start, end, span.text)

        return matches

    @staticmethod
    def phrase_matcher(terminology_list, text):
        matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
        patterns = [nlp.make_doc(txt) for txt in terminology_list]
        matcher.add("Phrase Matching", None, *patterns)

        doc = nlp(text)
        matches = matcher(doc)
        for match_id, start, end in matches:
            string_id = nlp.vocab.strings[match_id]
            span = doc[start:end]  # The matched span
            print(match_id, string_id, start, end, span.text)

        return matches

    @staticmethod
    def entity_matcher(patterns, text):
        ruler = EntityRuler(nlp)
        ruler.add_patterns(patterns)
        nlp.add_pipe(ruler)

        doc = nlp(text)
        print([(ent.text, ent.label_) for ent in doc.ents])
        return doc.ents
