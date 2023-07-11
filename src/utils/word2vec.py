import os
import pprint

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

from .constants import TOPICS


def train_for_label(path_to_clean_csv: str, label: str) -> Word2Vec:
    with open(path_to_clean_csv, 'r') as f:
        lines = f.readlines()[1:]  # skip header

    data: list[list[str]] = []
    for line in lines:
        # tweet_time, owner, tweet_text, university, owner_name, tweet_label = line.split(',')
        _, _, tweet_text, _, _, tweet_label = line.split(',')
        if tweet_text == '' or tweet_text is None:
            continue
        tweet_label = tweet_label[:-1]  # remove \n
        if tweet_label == label:
            tmp: list[str] = []
            for token in word_tokenize(tweet_text):
                tmp.append(token.lower())
            data.append(tmp)

    # skipgram model
    model = Word2Vec(data, min_count=1, vector_size=100, window=5, sg=1)

    return model


def train_for_all(path_to_clean_csv: str) -> Word2Vec:
    with open(path_to_clean_csv, 'r') as f:
        lines = f.readlines()[1:]  # skip header

    data: list[list[str]] = []
    for line in lines:
        tweet_text = line.split(',')[2]
        if tweet_text == '' or tweet_text is None:
            continue
        tmp: list[str] = []
        for token in word_tokenize(tweet_text):
            tmp.append(token.lower())
        data.append(tmp)

    # skipgram model
    model = Word2Vec(data, min_count=1, vector_size=100, window=5, sg=1)

    return model


def load_w2v_model(label: str) -> Word2Vec:
    if label != 'all' and label not in TOPICS.values():
        raise ValueError(f'Invalid label: {label}')
    path_to_model = f'../models/word2vec/{label}.npy'
    if not os.path.exists(path_to_model):
        raise FileNotFoundError(f'No model found for label: {label}. Please train first.')
    return Word2Vec.load(path_to_model)


def get_w2v_stats():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    words = [
        'سیاست',  # politics
        'ایران',  # Iran
        'گل',  # flower
        'ماشین'  # car
    ]
    all_model = load_w2v_model('all')

    img_location = '../stats/word2vec_similar.png'
    image = Image.new('RGB', (670, 650), color=(255, 255, 255))
    image.save(img_location)

    for i, word in enumerate(words):
        most_similar: list[tuple[str, float]] = all_model.wv.most_similar(word, topn=10)

        most_similar_str = pprint.pformat(most_similar)

        # font_file = f"{root}/fonts/Sahel.ttf"
        font_file = f"{root}/fonts/XBNiloofar.ttf"

        # load the font and image
        font = ImageFont.truetype(font_file, 18)
        image = Image.open(img_location)

        # start drawing on image
        text = 'کلمات مشابه برای کلمه ' + word + ':'
        text += '\n' + most_similar_str
        draw = ImageDraw.Draw(image)
        if i == 0:
            x, y = 5, 5
        elif i == 1:
            x, y = 355, 5
        elif i == 2:
            x, y = 5, 355
        else:
            x, y = 355, 355
        draw.text((x, y), text, (0, 0, 0), font=font)
        draw = ImageDraw.Draw(image)

        # save it
        image.save(img_location)
