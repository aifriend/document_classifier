class NGram:

    def __init__(self):
        self.n = 0
        self.tokens = []

    def add_gram(self, gram):
        self.tokens.append(gram)
        self.n = len(self.tokens)

    def __str__(self):
        if self.n == 2:
            return '(' + self.tokens[0].lemma + '; ' + self.tokens[1].lemma + ')'
        if self.n == 3:
            return '(' + self.tokens[0].text + '; ' + self.tokens[1].text + ', ' + self.tokens[2].text + ')'
        return 'unsupported ngram'
