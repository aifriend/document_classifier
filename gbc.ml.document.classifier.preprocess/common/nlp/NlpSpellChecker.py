from Levenshtein import StringMatcher


class NlpSpellChecker:

    @staticmethod
    def check(word, dictionary):
        if word is None or word is '' or str(word)[0].isnumeric():
            return []
        # if str(word)[0].isupper():
        #     return [' ']
        fl = str(word).lower()[0]
        sub_dict = list(filter(lambda x: x.startswith(fl), dictionary))
        similar = list(filter(lambda x: StringMatcher.distance(str(word).lower(), x) < 2, sub_dict))
        # similar = list(filter(lambda x: word.lower() == x, sub_dict))
        return similar

    @staticmethod
    def check_exact(word, dictionary):
        if word is None or word is '' or str(word)[0].isnumeric():
            return []
        x = str(word).lower()
        if x in dictionary:
            return [x]
        else:
            return []
