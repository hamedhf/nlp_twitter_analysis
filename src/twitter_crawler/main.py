from selenium import webdriver

from utils.crawling import crawl_tweets_by_username

driver = webdriver.Chrome()

usernames = [
    "DraiusX"
]

for username in usernames:
    tweets = crawl_tweets_by_username(driver, username, limit=20)
    print("Found {} tweets for username {}.".format(len(tweets), username))

driver.close()
driver.quit()
