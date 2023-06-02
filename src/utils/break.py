from hazm import word_tokenize, lemmatizer, stemmer


def complex_word_tokenizer(text) -> list[str]:
    """
    input text must be cleaned.
    """
    words = word_tokenize(text)
    lm = lemmatizer.Lemmatizer()
    words = [lm.lemmatize(word) for word in words]
    return words


def simple_word_tokenizer(text) -> list[str]:
    return word_tokenize(text)


# def sentence_tokenizer(text):
#     return sentence_tokenize(text)


if __name__ == "__main__":
    stemmer_object = stemmer.Stemmer()
    print(stemmer_object.stem('کتاب‌ها'))
    print(stemmer_object.stem('می‌روم'))

    lemmatizer_object = lemmatizer.Lemmatizer()
    print(lemmatizer_object.lemmatize('کتاب‌ها'))
    print(lemmatizer_object.lemmatize('می‌روم'))

    print(simple_word_tokenizer('من می‌روم کتاب‌ها را می‌خوانم'))
    print(complex_word_tokenizer('من می‌روم کتاب‌ها را می‌خوانم'))
