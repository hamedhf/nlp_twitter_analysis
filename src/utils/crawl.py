import csv
import sqlite3

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

IMPLICIT_WAIT_TIME = 5


def extract_tweets(driver: webdriver.Chrome, username) -> dict:
    tweet_elements = driver.find_elements(by=By.CSS_SELECTOR, value='article[data-testid="tweet"]')
    i = 0
    j = 0
    tweets = {}
    while i < len(tweet_elements):
        tweet_element = tweet_elements[i]
        try:
            tweet_owner = tweet_element.find_elements(
                by=By.XPATH,
                value='.//a[@role="link"]/div[@dir="ltr"]/span/../..'
            )[-1].get_attribute('href').split('/')[-1]
            if tweet_owner.lower().strip() != username.lower():
                print(f"Tweet owner {tweet_owner} does not match username {username}. Skipping.")
                i += 1
                continue
            tweet_id = tweet_element.find_element(by=By.CSS_SELECTOR, value='time').get_attribute('datetime')
            tweet_text = tweet_element.find_element(
                by=By.CSS_SELECTOR, value='div[data-testid="tweetText"]').get_attribute('innerText')
            print(f"Found tweet from {tweet_owner} with id {tweet_id} and text {tweet_text} for username {username}.")
            tweets[tweet_id] = tweet_text
            i += 1
            j = 0
        except NoSuchElementException:
            print(f"No tweet text for tweet element {i}")
            i += 1
            j = 0
        except (StaleElementReferenceException, WebDriverException):
            print(f"Stale or web driver exception for tweet element {i}.")
            if j > 5:
                print(f"Skipping tweet element {i}.")
                i += 1
                j = 0
                continue
            j += 1
            driver.implicitly_wait(IMPLICIT_WAIT_TIME * j)
    return tweets


def scroll_to_bottom(driver: webdriver.Chrome, body: WebElement) -> None:
    """
    Scroll to the bottom of the page.
    """
    body.send_keys(Keys.PAGE_DOWN)
    driver.implicitly_wait(IMPLICIT_WAIT_TIME)


def close_popups(driver: webdriver.Chrome) -> None:
    """
    Close popups.
    """
    popup_css = 'div[aria-label="Close"]'
    popups = driver.find_elements(by=By.CSS_SELECTOR, value=popup_css)
    popups.reverse()
    print(f"Found {len(popups)} popups.")
    for popup in popups:
        popup.click()


def user_not_found(driver: webdriver.Chrome) -> bool:
    error_xpath = './/span[contains(text(), "Hmm...this page doesn’t exist. Try searching for something else.")]'
    try:
        error = driver.find_element(by=By.XPATH, value=error_xpath)
        return True
    except NoSuchElementException:
        return False


def tweets_are_protected(driver: webdriver.Chrome) -> bool:
    # data-testid="empty_state_header_text"
    error_xpath = './/div[@data-testid="empty_state_header_text"]/span[contains(text(), "These Tweets are protected")]'
    try:
        error = driver.find_element(by=By.XPATH, value=error_xpath)
        return True
    except NoSuchElementException:
        return False


def crawl_tweets_by_username(driver: webdriver.Chrome, username, limit=100, click_cookies=False) -> dict:
    """
    Crawl tweets by username.
    """
    url = f"https://twitter.com/{username}"
    driver.get(url)

    if click_cookies:
        # Wait for the cookies button to appear and refuse them.
        cookies_button_xpath = './/div[@role="button"]/div/span/span[contains(text(), "Refuse")]'
        cookies_button = WebDriverWait(driver, 30).until(
            EC.element_to_be_clickable((By.XPATH, cookies_button_xpath))
        )
        cookies_button.click()
    else:
        driver.implicitly_wait(IMPLICIT_WAIT_TIME * 2)

    if user_not_found(driver):
        print(f"User {username} not found.")
        return {}
    elif tweets_are_protected(driver):
        print(f"Tweets for user {username} are protected.")
        return {}

    tweets = {}
    body = driver.find_element(by=By.CSS_SELECTOR, value='body')
    tolerance = 5
    while len(tweets) < limit and tolerance > 0:
        old_tweets_len = len(tweets)

        close_popups(driver)
        scroll_to_bottom(driver, body)
        tweets2 = extract_tweets(driver, username)
        tweets.update(tweets2)

        new_tweets_len = len(tweets)
        added_tweets_len = new_tweets_len - old_tweets_len
        print(f"Found {added_tweets_len} new tweets.")
        if added_tweets_len == 0:
            tolerance -= 1
            print("No new tweets found. Decreasing tolerance.")
        else:
            tolerance = 5

    return tweets


def create_unlabeled_table():
    conn = sqlite3.connect('../data/raw/unlabeled.db')
    cursor = conn.cursor()
    create_table_query = """ 
    CREATE TABLE IF NOT EXISTS TWEETS (
        tweet_time CHAR(255) NOT NULL,
        tweet_owner CHAR(255) NOT NULL,
        tweet_text CHAR(255) NOT NULL,
        owner_university CHAR(255) NOT NULL,
        owner_name CHAR(255) NOT NULL,
        UNIQUE(tweet_time, tweet_owner)
    );
    """
    cursor.execute(create_table_query)
    conn.commit()
    cursor.close()


def save_tweets(tweets: dict, username, university, name):
    conn = sqlite3.connect('../data/raw/unlabeled.db')
    cursor = conn.cursor()
    for tweet_time, tweet_text in tweets.items():
        tweet_text: str = tweet_text.strip().replace("'", '').replace('"', '')
        insert_query = "INSERT OR IGNORE INTO TWEETS (tweet_time, tweet_owner, tweet_text, owner_university, " + \
                       "owner_name) VALUES ('{}', '{}', '{}', '{}', '{}');".format(
                           tweet_time, username, tweet_text, university, name)
        print(insert_query)
        cursor.execute(insert_query)
        conn.commit()
    cursor.close()


def get_users(csv_path):
    users = []
    with open(csv_path, 'r') as csv_file:
        rows = list(csv.reader(csv_file, delimiter=','))
        for row in rows[1:]:
            username = row[0]
            university = row[1]
            name = row[2]
            users.append((username, university, name))

    return users
