from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize


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
