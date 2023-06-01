from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

IMPLICIT_WAIT_TIME = 3


def extract_tweets(driver: webdriver.Chrome) -> dict:
    tweet_elements = driver.find_elements(by=By.CSS_SELECTOR, value='article[data-testid="tweet"]')
    i = 0
    j = 0
    tweets = {}
    while i < len(tweet_elements):
        tweet_element = tweet_elements[i]
        try:
            tweet_id = tweet_element.find_element(by=By.CSS_SELECTOR, value='time').get_attribute('datetime')
            print(tweet_id)
            tweet_text = tweet_element.find_element(by=By.CSS_SELECTOR, value='div[data-testid="tweetText"]').text
            print(tweet_text)
            tweets[tweet_id] = tweet_text
            i += 1
        except NoSuchElementException:
            print(f"No tweet text for tweet element {i}")
            i += 1
        except StaleElementReferenceException:
            print(f"Stale element reference exception for tweet element {i}")
            if j > 5:
                print(f"Too many stale element reference exceptions for tweet element {i}. Skipping.")
                i += 1
                j = 0
                continue
            driver.implicitly_wait(IMPLICIT_WAIT_TIME)
            j += 1
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


def crawl_tweets_by_username(driver: webdriver.Chrome, username, limit=100) -> dict:
    """
    Crawl tweets by username.
    """
    url = f"https://twitter.com/{username}"
    driver.get(url)

    # Wait for the cookies button to appear and refuse them.
    cookies_button_xpath = './/div[@role="button"]/div/span/span[contains(text(), "Refuse")]'
    cookies_button = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.XPATH, cookies_button_xpath))
    )
    cookies_button.click()

    tweets = {}
    body = driver.find_element(by=By.CSS_SELECTOR, value='body')
    while len(tweets) < limit:
        old_tweets_len = len(tweets)

        close_popups(driver)
        scroll_to_bottom(driver, body)
        tweets2 = extract_tweets(driver)
        tweets.update(tweets2)

        new_tweets_len = len(tweets)
        added_tweets_len = new_tweets_len - old_tweets_len
        print(f"Found {added_tweets_len} new tweets.")
        if added_tweets_len == 0:
            print("No new tweets found. Stopping.")
            break

    return tweets
