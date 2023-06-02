import logging
import os
from datetime import datetime

import typer
from dotenv import load_dotenv
from selenium import webdriver

from utils.cleaning import clean_text
from utils.crawling import crawl_tweets_by_username, get_users, save_tweets, create_unlabeled_table
from utils.labeling import get_tweet_label, get_api_key, get_crawled_tweets

app = typer.Typer()
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


def check_necessary_files():
    if not os.path.exists('../data'):
        os.mkdir('../data')

    if not os.path.exists('../data/raw'):
        os.mkdir('../data/raw')

    if not os.path.exists('../data/clean'):
        os.mkdir('../data/clean')

    if not os.path.exists('../data/wordbroken'):
        os.mkdir('../data/wordbroken')

    if not os.path.exists('../data/sentencebroken'):
        os.mkdir('../data/sentencebroken')

    if not os.path.exists('../stats'):
        os.mkdir('../stats')

    if not os.path.exists('./users.csv'):
        raise FileNotFoundError("users.csv not found. Please create it.")

    if not os.path.exists('./.env'):
        raise FileNotFoundError(".env not found. Please create it like .env.example.")


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
        label = get_tweet_label(api_key, tweet_text).replace(',', ' ').replace('\n', ' ').replace('\r', ' ').strip()
        with open(labeled_csv_path, 'a') as f:
            f.write("{},{},{},{},{},{}\n".format(
                tweet_time, tweet_owner, tweet_text, owner_university, owner_name, label))


@app.command()
def example_labeling():
    api_key = get_api_key()
    persian_tweet = "امروز با بچه‌ها میخوایم بریم بیرون و بعدش بریم سینما"
    logger.info("Labeling example: {}".format(persian_tweet))
    label = get_tweet_label(api_key, persian_tweet)
    logger.info("Label: {}".format(label))


@app.command()
def clean_data(path_to_labeled_csv: str):
    logger.info("Cleaning data...")

    # check labeled.csv exists
    if not os.path.exists(path_to_labeled_csv):
        raise FileNotFoundError("labeled.csv not found. Please provide a valid csv file or run label_data first.")

    # open labeled.csv
    with open(path_to_labeled_csv, 'r') as f:
        lines = f.readlines()

    date = os.path.basename(path_to_labeled_csv).split('_')[-1]
    file_path = '../data/clean/cleaned_{}'.format(date)
    with open(file_path, 'w') as f:
        f.write("tweet_time,tweet_owner,tweet_text,owner_university,owner_name,label\n")

    for line in lines[1:]:
        tweet_time, tweet_owner, tweet_text, owner_university, owner_name, label = line.split(',')
        tweet_text = clean_text(tweet_text)
        with open(file_path, 'a') as f:
            f.write("{},{},{},{},{},{}".format(
                tweet_time, tweet_owner, tweet_text, owner_university, owner_name, label))

    logger.info("Cleaned data saved in {}".format(file_path))


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(current_dir)
    check_necessary_files()
    load_dotenv('./.env')
    DEBUG = os.getenv("DEBUG") in ['True', 'true', '1']
    logger = create_logger()
    app()
