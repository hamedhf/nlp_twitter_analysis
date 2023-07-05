from hazm import word_tokenize, lemmatizer, stemmer, sent_tokenize

from .constants import PAD_TOKEN


def complex_word_tokenizer(text) -> list[str]:
    """
    input text must be cleaned.
    """
    words = word_tokenize(text)
    lm = lemmatizer.Lemmatizer()
    words = [lm.lemmatize(word) for word in words]
    return words


def simple_word_tokenizer(text: str) -> list[str]:
    return word_tokenize(text)


def pad_list(input_list: list[str], max_length) -> list[str]:
    return input_list + [PAD_TOKEN] * (max_length - len(input_list))


def simple_sentence_tokenizer(text: str) -> list[str]:
    """
    input text must have punctuation.
    """
    sentences = sent_tokenize(text)
    return sentences


if __name__ == "__main__":
    stemmer_object = stemmer.Stemmer()
    print(stemmer_object.stem('کتاب‌ها'))  # noqa
    print(stemmer_object.stem('می‌روم'))

    lemmatizer_object = lemmatizer.Lemmatizer()
    print(lemmatizer_object.lemmatize('کتاب‌ها'))  # noqa
    print(lemmatizer_object.lemmatize('می‌روم'))

    print(simple_word_tokenizer('من می‌روم کتاب‌ها را می‌خوانم'))  # noqa
    print(complex_word_tokenizer('من می‌روم کتاب‌ها را می‌خوانم'))  # noqa

    print(simple_sentence_tokenizer('من می‌روم. کتاب‌ها را می‌خوانم.'))  # noqa
