class Token:

    def __init__(self):
        self.text = ''
        self.lemma = ''
        self.pos = ''
        self.ner = ''
        self.stop = False  ## removable as a stopword for the context
        self.real = True  ## is a real word (from dictionary)
        self.freq = 0.0
        self.vec = []
        self.spacy_token = None

    def set(self, text, lemma, pos, ner, stop, real, freq):
        self.text = text
        self.lemma = lemma
        self.pos = pos
        self.ner = ner
        self.stop = stop
        self.real = real
        self.freq = freq

    def set_spacy_token(self, token):
        # self.spacy_token = token
        self.text = token.text
        self.lemma = token.lemma_
        self.pos = token.pos_
        self.ner = token.ent_type_
        self.stop = token.is_stop
        if token.has_vector:
            self.vec = token.vector
        return self

    def __str__(self):
        return self.text + ' ' + self.lemma + ' ' + self.pos + ' ' + self.ner + ' ' + str(self.stop)
