import logging
import os

import typer
from dotenv import load_dotenv
from selenium import webdriver

from utils.crawling import crawl_tweets_by_username, get_users, save_tweets

DEBUG = True

# logger setup
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s: %(levelname)s/%(module)s-%(funcName)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(level=logging.DEBUG if DEBUG else logging.WARNING)

app = typer.Typer()


def check_necessary_files():
    if not os.path.exists('../data'):
        os.mkdir('../data')

    if not os.path.exists('../data/raw'):
        os.mkdir('../data/raw')

    if not os.path.exists('../data/clean'):
        os.mkdir('../data/clean')

    if not os.path.exists('./users.csv'):
        raise FileNotFoundError("users.csv not found. Please create it.")

    if not os.path.exists('./.env'):
        raise FileNotFoundError(".env not found. Please create it like .env.example.")


@app.command()
def scrape_twitter():
    logger.info("Scraping Twitter...")
    users = get_users('./users.csv')
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
    pass


@app.command()
def clean_data():
    pass


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(current_dir)
    check_necessary_files()
    load_dotenv('.env')
    app()
