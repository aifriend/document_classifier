from common.model.Token import Token


class Sentence:

    def __init__(self):
        self.tokens = []

    def add_token(self, token):
        self.tokens.append(token)

    def add_spacy_token(self, token):
        self.tokens.append(Token().set_spacy_token(token))
