import re
import string

import nltk
from hazm import Normalizer, word_tokenize, stopwords_list, stemmer, lemmatizer
from nltk.corpus import stopwords


def remove_emoji(text):
    # Happy
    grin = 'خنده'
    laugh = 'خنده'
    happy = 'خوشحال'
    text = re.sub(":D", grin, text)
    text = re.sub(" (x|X)D", laugh, text)
    text = re.sub(":\)+", happy, text)

    # Sad
    sad = 'ناراحت'
    annoyed = 'رنجیده'
    text = re.sub(":\(+", sad, text)
    text = re.sub("-_+-", annoyed, text)
    return text


def remove_url(text):
    text = re.sub("https?:\/\/t.co\/[A-Za-z0-9]*", '', text)
    return text


def remove_punc(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


def remove_stopwords(text):
    # TODO: Implement this
    # link: https://medium.com/analytics-vidhya/a-simple-yet-effective-way-of-text-cleaning-using-nltk-4f90a8ff21d4
    # text_data = [wl.lemmatize(word) for word in text_data if not word in set(stopwords.words('english'))]
    pass


def remove_duplicate_spaces(text):
    _text = " ".join(text.split())
    return _text


def clean_text(text):
    _text = remove_duplicate_spaces(
        remove_punc(
            remove_url(
                remove_emoji(text)
            )
        )
    )
    normalizer = Normalizer()
    _text = normalizer.normalize(_text)
    return _text


def persian_tokenizer(text) -> list[str]:
    return word_tokenize(text)


if __name__ == "__main__":
    uncleaned_persian_text = "امروز با بچه‌ها میخوایم بریم بیرون و بععععدش بریم سینما :D https://t.co/1234567890"
    cleaned_text = clean_text(uncleaned_persian_text)
    print(cleaned_text)
    print(persian_tokenizer(cleaned_text))

    print(stopwords_list())

    stemmer_object = stemmer.Stemmer()
    print(stemmer_object.stem('کتاب‌ها'))
    print(stemmer_object.stem('می‌روم'))

    lemmatizer_object = lemmatizer.Lemmatizer()
    print(lemmatizer_object.lemmatize('کتاب‌ها'))
    print(lemmatizer_object.lemmatize('می‌روم'))

    nltk.download('stopwords')
    print(stopwords.words('english'))

    # download model from: https://github.com/roshan-research/hazm
    # tagger = POSTagger(model='resources/pos_tagger.model')
    # tagger.tag(word_tokenize('ما بسیار کتاب می‌خوانیم'))
