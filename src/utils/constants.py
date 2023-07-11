import os

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
PAD_TOKEN_OLD = 'PAD'
PAD_TOKEN = '<PAD>'

PRESELECTED_LABELS = ["politics_and_current_affairs", "education_and_learning", "environment_and_sustainability",
                      "home_and_garden", "weather_and_seasons"]


def get_api_base_url():
    api_base_url = os.getenv("OPENAI_API_BASE_URL")
    if api_base_url is None:
        raise ValueError("OPENAI_API_BASE_URL not found in .env file.")
    else:
        return api_base_url


def get_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("OPENAI_API_KEY not found in .env file.")
    else:
        return api_key
