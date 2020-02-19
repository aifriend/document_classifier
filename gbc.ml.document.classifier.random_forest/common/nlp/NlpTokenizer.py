class NlpTokenizer:

    @staticmethod
    def sentence_tokenizer(nlp, text):
        doc = nlp(text)
        for i, token in enumerate(doc.sents):
            print('-->Sentence %d: %s' % (i, token.text))

        return doc

    @staticmethod
    def words_tokenizer(nlp, text):
        doc = nlp(text)
        for token in doc:
            print(token.text,
                  token.lemma_,
                  token.pos_,
                  token.tag_,
                  token.dep_,
                  token.shape_,
                  token.is_alpha,
                  token.is_stop)

        return doc
