import os


def check_necessary_files():
    if not os.path.exists('../data'):
        os.mkdir('../data')

    if not os.path.exists('../data/raw'):
        os.mkdir('../data/raw')

    if not os.path.exists('../data/clean'):
        os.mkdir('../data/clean')

    if not os.path.exists('../data/augment'):
        os.mkdir('../data/augment')

    if not os.path.exists('../data/wordbroken'):
        os.mkdir('../data/wordbroken')

    if not os.path.exists('../data/sentencebroken'):
        os.mkdir('../data/sentencebroken')

    if not os.path.exists('../data/split'):
        os.mkdir('../data/split')

    if not os.path.exists('../stats'):
        os.mkdir('../stats')

    if not os.path.exists('../models'):
        os.mkdir('../models')

    if not os.path.exists('../models/word2vec'):
        os.mkdir('../models/word2vec')

    if not os.path.exists('../models/parsbert'):
        os.mkdir('../models/parsbert')

    if not os.path.exists('../models/huggingface_cache'):
        os.mkdir('../models/huggingface_cache')

    if not os.path.exists('../logs'):
        os.mkdir('../logs')

    if not os.path.exists('./users.csv'):
        raise FileNotFoundError("users.csv not found. Please create it.")

    if not os.path.exists('./.env'):
        raise FileNotFoundError(".env not found. Please create it like .env.example.")
