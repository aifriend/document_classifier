from common.model.SpacyModel import SpacyModel
from common.model.Token import Token
from common.nlp.NlpSpellChecker import NlpSpellChecker


class NlpStopwords:

    @staticmethod
    def stopwords_removal(nlp, text):
        doc = nlp(text)

        print([token for token in doc])
        tokens = [token for token in doc if not token.is_stop]
        print(tokens)

        return tokens

    @staticmethod
    def other_stopwords_removal(nlp, text):
        doc = nlp(text)

        tokens = [token for token in doc if not token.is_stop]
        for ent in doc.ents:
            print(ent)
        print(tokens)

        return tokens

    @staticmethod
    def clean_token(conf, token):
        t = Token()
        t.set_spacy_token(token)
        dictionary = SpacyModel.getInstance().getDictionary(conf.dictionary)
        spelled = NlpSpellChecker.check_exact(token.lemma_, dictionary)

        # if '+93' in token.text:
        #     print(token.text, str(token.pos_), str(token.ent_type_))

        if str(token.pos_) == 'NOUN':  # \
            # or str(token.pos_) == 'PROPN':
            t.stop = False

        if str(token.ent_type_) == 'PER' \
                or token.like_num \
                or token.like_url \
                or token.like_email \
                or token.is_quote \
                or token.is_bracket \
                or token.is_space \
                or token.is_right_punct \
                or token.is_left_punct \
                or token.is_punct \
                or token.is_digit \
                or token.is_currency \
                or len(token.text) <= 3 \
                or len(spelled) == 0:
            t.stop = True

        if str(token.ent_type_) == 'MISC' \
                or str(token.ent_type_) == 'ORG' \
                or len(token.text) > 3 and len(spelled) > 0:
            t.stop = False

        if str(token.pos_) == 'SPACE' \
                or str(token.pos_) == 'NUM' \
                or str(token.pos_) == 'DET' \
                or str(token.pos_) == 'CONJ' \
                or str(token.pos_) == 'SCONJ' \
                or str(token.pos_) == 'PUNCT' \
                or '/' in token.text \
                or '' in token.text \
                or len(token.text) <= 2:
            t.stop = True
            # or '/' in token.text \???

        return t
