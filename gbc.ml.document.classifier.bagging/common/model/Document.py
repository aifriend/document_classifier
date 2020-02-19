class Document:

    def __init__(self, kind='', text=''):
        self.sentences = []
        self.kind = kind
        self.text = text
        self.grams = []
        self.bi_grams = []
        self.tri_grams = []
        self.path = ''

    def get_grams_as_text(self):
        return ' '.join(list(self.grams))

    def add_sentence(self, sentence):
        self.sentences.append(sentence)

    def add_gram(self, word):
        self.grams.append(word)

    def add_bi_gram(self, bi_gram):
        self.bi_grams.append(bi_gram)

    def add_tri_gram(self, tri_gram):
        self.tri_grams.append(tri_gram)
