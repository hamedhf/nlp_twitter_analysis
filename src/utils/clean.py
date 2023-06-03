import re
import string

import nltk
from hazm import Normalizer, stopwords_list
from nltk.corpus import stopwords


def remove_emoji(text):
    # Happy
    grin = 'خنده'
    laugh = 'خنده'
    happy = 'خوشحال'
    _text = re.sub(":D", grin, text)
    _text = re.sub(" (x|X)D", laugh, _text)
    _text = re.sub(":\)+", happy, _text)

    # Sad
    sad = 'ناراحت'
    annoyed = 'رنجیده'
    _text = re.sub(":\(+", sad, _text)
    _text = re.sub("-_+-", annoyed, _text)
    return _text


def remove_url(text):
    _text = re.sub(r"https?:\S+", '', text)
    return _text


def remove_punc(text):
    _text = text.translate(str.maketrans('', '', string.punctuation))
    persian_virgol = '،'  # noqa
    _text = _text.replace(persian_virgol, ' ')
    return _text


def remove_stopwords(text):
    # TODO: Check if this is a good idea or not.
    # link: https://medium.com/analytics-vidhya/a-simple-yet-effective-way-of-text-cleaning-using-nltk-4f90a8ff21d4
    # text_data = [wl.lemmatize(word) for word in text_data if not word in set(stopwords.words('english'))]
    pass


def remove_numbers(text):
    _text = re.sub(r'\d+', '', text)
    return _text


def remove_hashtags(text):
    _text = re.sub(r'#\S+', '', text)
    return _text


def remove_mentions(text):
    _text = re.sub(r'@\S+', '', text)
    return _text


def remove_duplicate_spaces(text):
    _text = " ".join(text.split())
    return _text


def clean_text(text) -> tuple[str, str]:
    _punc_text = remove_duplicate_spaces(
        remove_url(
            remove_emoji(text)
        )
    )

    _text = remove_punc(_punc_text)
    normalizer = Normalizer()
    _text = normalizer.normalize(_text)

    return _text, _punc_text


if __name__ == "__main__":
    uncleaned_persian_text = "امروز با بچه‌ها میخوایم بریم بیرون و بععععدش بریم سینما :D https://t.co/1234567890"
    cleaned_text = clean_text(uncleaned_persian_text)
    print(cleaned_text)

    print(stopwords_list())
    nltk.download('stopwords')
    print(stopwords.words('english'))

    print(remove_url('https://github.com/roshan-research/hazm hi there'))
    print(remove_url('https hi there'))

    # download model from: https://github.com/roshan-research/hazm
    # tagger = POSTagger(model='resources/pos_tagger.model')
    # tagger.tag(word_tokenize('ما بسیار کتاب می‌خوانیم'))
