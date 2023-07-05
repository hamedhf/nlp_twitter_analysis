import sqlite3

import openai

from .constants import TOPICS


def get_clean_label(input_label: str) -> str:
    for value in TOPICS.values():
        if value in input_label:
            label = value
            break
    else:
        label = 'other'
    return label


def get_tweet_label(api_key: str, api_base: str, tweet: str) -> str:
    openai.api_key = api_key
    openai.api_base = api_base
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
        label = get_clean_label(label)
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
