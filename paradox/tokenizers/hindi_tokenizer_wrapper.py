from paradox.tokenizers.hindi_tokenizer import Tokenizer


def hindi_tokenize(s):
    tokenizer = Tokenizer(s)
    tokenizer.tokenize()
    return tokenizer.tokens
