import logging
import os
import random
from datetime import datetime

import torch
import typer
from dotenv import load_dotenv
from selenium import webdriver

from utils.augment import get_tweet_count_per_label, augment_label, append_augmented_data
from utils.basic import check_necessary_files, latex_pdf_report
from utils.clean import clean_text, get_clean_label, translate_english_to_persian
from utils.constants import get_api_key, get_api_base_url, TOPICS
from utils.crawl import crawl_tweets_by_username, get_users, save_tweets, create_unlabeled_table
from utils.gpt2 import train_gpt2, prepare_language_model_dataset, gpt2_generator
from utils.label import get_tweet_label, get_crawled_tweets
from utils.parsbert import train_parsbert
from utils.segment import simple_word_tokenizer, pad_list, simple_sentence_tokenizer
from utils.split import prepare_dataset
from utils.stats import (
    get_tweet_count,
    get_segment_count,
    get_unique_word_count,
    write_dict_to_csv,
    top_ten_frequent_word_per_label,
    get_plot, write_dict_to_csv2
)
from utils.word2vec import train_for_label, train_for_all, load_w2v_model, get_w2v_stats

app = typer.Typer()
current_dir = None
parr_dir = None
huggingface_cache_dir = None
logger = None
DEBUG = None


def create_logger():
    new_logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s: %(levelname)s/%(module)s-%(funcName)s] %(message)s")
    handler.setFormatter(formatter)
    new_logger.addHandler(handler)
    new_logger.setLevel(level=logging.DEBUG if DEBUG else logging.WARNING)
    return new_logger


@app.command()
def scrape_twitter():
    logger.info("Scraping Twitter...")
    users = get_users('./users.csv')
    create_unlabeled_table()
    driver = webdriver.Chrome()
    for i, user in enumerate(users):
        username = user[0].strip().lower()
        university = user[1].strip()
        name = user[2].strip()
        tweets = crawl_tweets_by_username(driver, username, limit=100, click_cookies=(i == 0))
        print("Found {} tweets for username {} and saving them...".format(len(tweets), username))
        save_tweets(tweets, username, university, name)
    driver.close()
    driver.quit()


@app.command()
def label_data():
    api_key = get_api_key()
    api_base_url = get_api_base_url()

    # check unlabeled.db exists
    if not os.path.exists('../data/raw/unlabeled.db'):
        raise FileNotFoundError("unlabeled.db not found in ../data/raw. Please run scrape_twitter first.")

    # load data from unlabeled.db
    tweets = get_crawled_tweets()

    # create labeled.csv
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    labeled_csv_path = "../data/raw/labeled_{}.csv".format(current_time)
    with open(labeled_csv_path, 'w') as f:
        f.write("tweet_time,tweet_owner,tweet_text,owner_university,owner_name,label\n")

    # label tweets
    for tweet in tweets:
        tweet_time = tweet[0].strip()
        tweet_owner = tweet[1].strip()
        tweet_text: str = tweet[2].replace('\n', ' ').replace('\r', ' ').replace(',', ' ').strip()
        owner_university = tweet[3].strip()
        owner_name = tweet[4].strip()
        label = get_tweet_label(
            api_key, api_base_url, tweet_text).replace(',', ' ').replace('\n', ' ').replace('\r', ' ').strip()
        with open(labeled_csv_path, 'a') as f:
            f.write("{},{},{},{},{},{}\n".format(
                tweet_time, tweet_owner, tweet_text, owner_university, owner_name, label))


@app.command()
def example_labeling():
    api_key = get_api_key()
    api_base_url = get_api_base_url()
    persian_tweet = "امروز با بچه‌ها میخوایم بریم بیرون و بعدش بریم سینما"  # noqa
    logger.info("Labeling example: {}".format(persian_tweet))
    label = get_tweet_label(api_key, api_base_url, persian_tweet)
    logger.info("Label: {}".format(label))


@app.command()
def clean_data(path_to_labeled_csv: str, skip: bool = False, skip_count: int = 1):
    logger.info("Cleaning data...")

    # check labeled.csv exists
    if not os.path.exists(path_to_labeled_csv):
        raise FileNotFoundError("labeled.csv not found. Please provide a valid csv file or run label_data first.")

    # open labeled.csv
    with open(path_to_labeled_csv, 'r') as f:
        lines = f.readlines()

    date = os.path.basename(path_to_labeled_csv).split('_')[-1]
    clean_file_path = '../data/clean/cleaned_{}'.format(date)
    punc_file_path = '../data/clean/punc_{}'.format(date)
    api_key = get_api_key()
    api_base_url = get_api_base_url()

    if not skip:
        with open(clean_file_path, 'w') as f:
            f.write("tweet_time,tweet_owner,tweet_text,owner_university,owner_name,label\n")
        with open(punc_file_path, 'w') as f:
            f.write("tweet_time,tweet_owner,tweet_text,owner_university,owner_name,label\n")

    for line in lines[skip_count:]:
        tweet_time, tweet_owner, tweet_text, owner_university, owner_name, label = line.split(',')

        if tweet_text == '':
            continue

        tweet_text = translate_english_to_persian(api_key, api_base_url, tweet_text)
        tweet_text, tweet_text_punc = clean_text(tweet_text)
        label = get_clean_label(label)

        with open(clean_file_path, 'a') as f:
            f.write("{},{},{},{},{},{}\n".format(
                tweet_time, tweet_owner, tweet_text, owner_university, owner_name, label))
        with open(punc_file_path, 'a') as f:
            f.write("{},{},{},{},{},{}\n".format(
                tweet_time, tweet_owner, tweet_text_punc, owner_university, owner_name, label))

    logger.info("Cleaned data saved in {}".format(clean_file_path))
    logger.info("Cleaned data with punctuation saved in {}".format(punc_file_path))


@app.command()
def break_by_word(path_to_clean_csv: str):
    logger.info("Breaking data by word...")
    # check clean.csv exists
    if not os.path.exists(path_to_clean_csv):
        raise FileNotFoundError("clean.csv not found. Please provide a valid csv file or run clean_data first.")
    with open(path_to_clean_csv, 'r') as f:
        lines = f.readlines()
    date = os.path.basename(path_to_clean_csv).split('_')[-1]
    word_file_path = '../data/wordbroken/wordbroken_{}'.format(date)
    tweets: list[list[str]] = [
        simple_word_tokenizer(line.split(',')[2]) for line in lines[1:]
    ]
    max_len = max([len(tweet) for tweet in tweets])
    tweets = [pad_list(tweet, max_len) for tweet in tweets]
    with open(word_file_path, 'w') as f:
        for tweet in tweets:
            f.write("{}\n".format(','.join(tweet)))


@app.command()
def break_by_sentence(path_to_punc_csv: str):
    logger.info("Breaking data by sentence...")
    # check punc.csv exists
    if not os.path.exists(path_to_punc_csv):
        raise FileNotFoundError("punc.csv not found. Please provide a valid csv file or run clean_data first.")
    with open(path_to_punc_csv, 'r') as f:
        lines = f.readlines()
    date = os.path.basename(path_to_punc_csv).split('_')[-1]
    sentence_file_path = '../data/sentencebroken/sentencebroken_{}'.format(date)
    tweets: list[list[str]] = [
        simple_sentence_tokenizer(line.split(',')[2]) for line in lines[1:]
    ]
    max_len = max([len(tweet) for tweet in tweets])
    tweets = [pad_list(tweet, max_len) for tweet in tweets]
    with open(sentence_file_path, 'w') as f:
        for tweet in tweets:
            f.write("{}\n".format(','.join(tweet)))


@app.command()
def get_stats(file_timestamp: str):
    path_to_clean_csv = '../data/clean/cleaned_{}.csv'.format(file_timestamp)
    path_to_punc_csv = '../data/clean/punc_{}.csv'.format(file_timestamp)
    path_to_word_csv = '../data/wordbroken/wordbroken_{}.csv'.format(file_timestamp)
    path_to_sentence_csv = '../data/sentencebroken/sentencebroken_{}.csv'.format(file_timestamp)

    # check clean.csv exists
    if not os.path.exists(path_to_clean_csv):
        raise FileNotFoundError(
            f"{path_to_clean_csv} not found. Please provide a valid csv file or run clean_data first.")

    # check punc.csv exists
    if not os.path.exists(path_to_punc_csv):
        raise FileNotFoundError(
            f"{path_to_punc_csv} not found. Please provide a valid csv file or run clean_data first.")

    # check wordbroken.csv exists
    if not os.path.exists(path_to_word_csv):
        raise FileNotFoundError(
            f"{path_to_word_csv} not found. Please provide a valid csv file or run break_by_word first.")

    # check sentencebroken.csv exists
    if not os.path.exists(path_to_sentence_csv):
        raise FileNotFoundError(
            f"{path_to_sentence_csv} not found. Please provide a valid csv file or run break_by_sentence first.")

    # get stats
    tweet_count = get_tweet_count(path_to_clean_csv)
    word_count = get_segment_count(path_to_word_csv)
    sentence_count = get_segment_count(path_to_sentence_csv)
    unique_word_count = get_unique_word_count(path_to_word_csv)

    logger.info("Tweet count: {}".format(tweet_count))
    logger.info("Word count: {}".format(word_count))
    logger.info("Sentence count: {}".format(sentence_count))
    logger.info("Unique word count: {}".format(unique_word_count))

    dictionary = {
        'tweet-count': tweet_count,
        'word-count': word_count,
        'sentence-count': sentence_count,
        'unique-word-count': unique_word_count
    }
    write_dict_to_csv(dictionary, '../stats/stats_{}.csv'.format(file_timestamp))

    top_ten_frequent = top_ten_frequent_word_per_label(path_to_word_csv, path_to_clean_csv)
    get_plot(file_timestamp, top_ten_frequent)


@app.command()
def generate_pdf_report(file_timestamp: str):
    latex_pdf_report(1, file_timestamp)


@app.command()
def augment_data(path_to_clean_csv: str, min_tweet_count_per_label: int = 200):
    logger.info("Augmenting data...")

    # check clean.csv exists
    if not os.path.exists(path_to_clean_csv):
        raise FileNotFoundError("clean.csv not found. Please provide a valid csv file or run clean_data first.")

    with open(path_to_clean_csv, 'r') as f:
        lines = f.readlines()[1:]  # skip header

    date = os.path.basename(path_to_clean_csv).split('_')[-1]
    augmented_file_path = '../data/augment/augmented_{}'.format(date)
    with open(augmented_file_path, 'w') as f:
        f.write("tweet_time,tweet_owner,tweet_text,owner_university,owner_name,label\n")
    append_augmented_data(augmented_file_path, lines)

    counts = get_tweet_count_per_label(path_to_clean_csv)
    print(f"cleaned data counts: {counts}")

    for label, count in counts.items():
        if count < min_tweet_count_per_label:
            logger.info("Augmenting label: {}".format(label))
            _ = augment_label(label, min_tweet_count_per_label - count, augmented_file_path, temperature=0.6)

    # shuffle lines
    with open(augmented_file_path, 'r') as f:
        lines = f.readlines()[1:]
        random.shuffle(lines)
    with open(augmented_file_path, 'w') as f:
        f.write("tweet_time,tweet_owner,tweet_text,owner_university,owner_name,label\n")
    append_augmented_data(augmented_file_path, lines)

    counts = get_tweet_count_per_label(augmented_file_path)
    print(f"augmented data counts: {counts}")
    # save counts to csv
    write_dict_to_csv2(counts, '../stats/augmented_counts.csv', 'label,tweet count')


@app.command()
def train_word2vec_label(path_to_clean_csv: str, label: str):
    # check clean.csv exists
    if not os.path.exists(path_to_clean_csv):
        raise FileNotFoundError("clean.csv not found. Please provide a valid csv file or run clean_data first.")
    if label not in TOPICS.values():
        raise ValueError(f"Invalid label. Please provide a valid label. Valid labels are: {TOPICS.values()}")
    logger.info(f"Training word2vec model for label: {label}")
    model = train_for_label(path_to_clean_csv, label)
    model.save(f'../models/word2vec/{label}.npy')
    logger.info(f"Model saved at {parr_dir}/models/word2vec/{label}.npy")


@app.command()
def train_word2vec_preselected(path_to_clean_csv: str):
    # check clean.csv exists
    if not os.path.exists(path_to_clean_csv):
        raise FileNotFoundError("clean.csv not found. Please provide a valid csv file or run clean_data first.")
    preselected_labels = ["politics_and_current_affairs", "education_and_learning", "environment_and_sustainability",
                          "home_and_garden", "weather_and_seasons"]
    for label in preselected_labels:
        if label not in TOPICS.values():
            raise ValueError(
                f"Invalid preselected labels. Please provide a valid label. Valid labels are: {TOPICS.values()}")
    for label in preselected_labels:
        logger.info(f"Training word2vec model for label: {label}")
        model = train_for_label(path_to_clean_csv, label)
        model.save(f'../models/word2vec/{label}.npy')
        logger.info(f"Model saved at {parr_dir}/models/word2vec/{label}.npy")


@app.command()
def train_word2vec_all(path_to_clean_csv: str):
    # check clean.csv exists
    if not os.path.exists(path_to_clean_csv):
        raise FileNotFoundError("clean.csv not found. Please provide a valid csv file or run clean_data first.")
    logger.info(f"Training word2vec model for all labels")
    model = train_for_all(path_to_clean_csv)
    model.save(f'../models/word2vec/all.npy')
    logger.info(f"Model saved at {parr_dir}/models/word2vec/all.npy")


@app.command()
def get_most_similar_words(label: str, word: str, topn: int = 10):
    model = load_w2v_model(label)
    logger.info(f"Most similar words to {word} in {label} are:")
    for word, similarity in model.wv.most_similar(word, topn=topn):
        logger.info(f"{similarity} \t {word}")


@app.command()
def get_word2vec_stats():
    get_w2v_stats()


@app.command()
def fine_tune_gpt2(path_to_augmented_csv: str, desired_label: str = None):
    logger.info("Fine tuning gpt2...")
    if not os.path.exists(path_to_augmented_csv):
        raise FileNotFoundError("augmented.csv not found. Please provide a valid csv file or run augment_data first.")
    csv_files = os.listdir("../data/languagemodel")
    for label in TOPICS.values():
        if f"{label}.csv" not in csv_files:
            logger.info("Preparing dataset for language modeling.")
            prepare_language_model_dataset(path_to_augmented_csv)
            break
    else:
        logger.info("Dataset already prepared. Skipping preparation step.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if desired_label is None:
        desired_labels = ["politics_and_current_affairs", "education_and_learning", "environment_and_sustainability",
                          "home_and_garden", "weather_and_seasons"]
    else:
        if desired_label not in TOPICS.values():
            raise ValueError(f"Invalid label. Please provide a valid label. Valid labels are: {TOPICS.values()}")
        desired_labels = [desired_label]

    for label in desired_labels:
        logger.info(f"Fine tuning gpt2 for label: {label}")
        train_gpt2(label, device)


@app.command()
def complete_prompt_gpt2(prompt: str, label: str):
    if label not in TOPICS.values():
        raise ValueError(f"Invalid label. Please provide a valid label. Valid labels are: {TOPICS.values()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info(f"Completing prompt: {prompt}")
    generated_outputs = gpt2_generator(prompt, label, device)
    logger.info(f"Generated outputs:")
    for output in generated_outputs:
        logger.info(f"{output}")


@app.command()
def fine_tune_parsbert(path_to_augmented_csv: str):
    logger.info("Fine tuning parsbert...")
    if not os.path.exists(path_to_augmented_csv):
        raise FileNotFoundError("augmented.csv not found. Please provide a valid csv file or run augment_data first.")
    csv_files = os.listdir("../data/split")
    if 'test.csv' not in csv_files or 'train.csv' not in csv_files or 'validation.csv' not in csv_files:
        prepare_dataset(path_to_augmented_csv)
    else:
        logger.info("Dataset already prepared. Skipping preparation step.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    train_parsbert(device)


@app.command()
def generate_final_pdf_report(file_timestamp: str = "2023-06-02-10-27-57"):
    latex_pdf_report(2, file_timestamp)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.realpath(__file__))
    parr_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    huggingface_cache_dir = parr_dir + '/models/huggingface_cache'
    os.chdir(current_dir)
    check_necessary_files()
    load_dotenv('./.env')
    DEBUG = os.getenv("DEBUG") in ['True', 'true', '1']
    logger = create_logger()
    app()
