import os
import sqlite3

import openai

TOPICS = {
    0: 'politics_and_current_affairs',
    1: 'entertainment_and_pop_culture',
    2: 'sports_and_athletics',
    3: 'technology_and_innovation',
    4: 'science_and_discovery',
    5: 'health_and_wellness',
    6: 'business_and_finance',
    7: 'travel_and_adventure',
    8: 'food_and_cooking',
    9: 'fashion_and_style',
    10: 'environment_and_sustainability',
    11: 'education_and_learning',
    12: 'social_issues_and_activism',
    13: 'inspirational_and_motivational',
    14: 'funny_and_humorous',
    15: 'art_and_design',
    16: 'books_and_literature',
    17: 'religion_and_spirituality',
    18: 'family_and_parenting',
    19: 'gaming',
    20: 'beauty_and_cosmetics',
    21: 'home_and_garden',
    22: 'automotive',
    23: 'pets_and_animals',
    24: 'weather_and_seasons',
    25: 'other'
}


def get_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("OPENAI_API_KEY not found in .env file.")
    else:
        return api_key


def get_tweet_label(api_key: str, tweet: str):
    openai.api_key = api_key
    openai.api_base = 'https://api.pawan.krd/v1'
    topics = list(TOPICS.values())
    messages = [
        {
            "role": "system",
            "content": f"Classify the topic of the future tweet into only one of the following categories: {', '.join(topics)}. some of these tweets are in slang persian language. please try to understand them. Just type the topic and nothing else."
        },
        {"role": "user", "content": f"Tweet: {tweet}"},

    ]
    print("*" * 100)
    print(messages)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.2,
        log_level="info"
    )
    print(response)
    print("*" * 100)

    try:
        label = str(response['choices'][0]['message']['content']).strip()
    except KeyError:
        print("KeyError occurred. Setting label to 'unknown'.")
        label = 'unknown'
    return label


def get_crawled_tweets():
    conn = sqlite3.connect('../data/raw/unlabeled.db')
    c = conn.cursor()
    c.execute("SELECT * FROM TWEETS")
    tweets = c.fetchall()
    conn.close()
    return tweets
