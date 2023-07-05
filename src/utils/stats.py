import matplotlib.pyplot as plt
from bidi import algorithm as bidialg
from hazm import lemmatizer

from .constants import TOPICS, PAD_TOKEN
from .label import get_clean_label


def get_tweet_count(path_to_clean_csv: str):
    with open(path_to_clean_csv, 'r') as f:
        lines = f.readlines()

    return len(lines) - 1


def get_segment_count(path_to_segment_csv: str):
    with open(path_to_segment_csv, 'r') as f:
        lines = f.readlines()

    # ignore PAD
    count = 0
    for line in lines:
        line = line[:-1]
        segments = line.split(',')
        for segment in segments:
            if segment != PAD_TOKEN:
                count += 1

    return count


def get_unique_word_count(path_to_word_csv: str):
    with open(path_to_word_csv, 'r') as f:
        lines = f.readlines()

    words = set()
    lm = lemmatizer.Lemmatizer()
    for line in lines:
        line = line[:-1]
        segments = line.split(',')
        for segment in segments:
            if segment != PAD_TOKEN:
                words.add(lm.lemmatize(segment))

    return len(words)


def top_ten_frequent_word_per_label(path_to_word_csv: str, path_to_clean_csv: str) -> dict:
    dictionary = {
        'unknown': {}
    }
    for topic in TOPICS.values():
        dictionary[topic] = {}

    with open(path_to_word_csv, 'r') as f:
        lines = f.readlines()

    with open(path_to_clean_csv, 'r') as f:
        clean_lines = f.readlines()[1:]

    lm = lemmatizer.Lemmatizer()
    for line, clean_line in zip(lines, clean_lines):
        # remove \n
        line = line[:-1]
        segments = line.split(',')
        label = get_clean_label(clean_line.split(',')[-1])
        for segment in segments:
            if segment != PAD_TOKEN:
                word = lm.lemmatize(segment)
                if word in dictionary[label]:
                    dictionary[label][word] += 1
                else:
                    dictionary[label][word] = 1

    for topic in TOPICS.values():
        dictionary[topic] = dict(sorted(dictionary[topic].items(), key=lambda item: item[1], reverse=True)[:10])

    return dictionary


def tf_idf_per_word(path_to_word_csv: str, path_to_clean_csv: str) -> dict:
    with open(path_to_word_csv, 'r') as f:
        lines = f.readlines()

    with open(path_to_clean_csv, 'r') as f:
        clean_lines = f.readlines()[1:]

    lm = lemmatizer.Lemmatizer()
    dictionary = {}
    for line, clean_line in zip(lines, clean_lines):
        # remove \n
        line = line[:-1]
        segments = line.split(',')
        label = get_clean_label(clean_line.split(',')[-1])
        for segment in segments:
            if segment != PAD_TOKEN:
                word = lm.lemmatize(segment)
                if word in dictionary:
                    if label in dictionary[word]:
                        dictionary[word][label] += 1
                    else:
                        dictionary[word][label] = 1
                else:
                    dictionary[word] = {
                        label: 1
                    }


def get_plot(file_timestamp: str, freq_dict: dict):
    path_to_plot = f'../stats/plot_{file_timestamp}.png'

    plt.figure(figsize=(15, 15))
    for topic_number, topic in TOPICS.items():
        for word, freq in freq_dict[topic].items():
            plt.scatter(topic_number, freq, marker='x', color='red')
            text = bidialg.get_display(word)
            plt.text(topic_number + .03, freq + .03, text, fontsize=9)

    plt.xlabel('Topics')
    plt.ylabel('Word count')
    plt.savefig(path_to_plot)


def write_dict_to_csv(dictionary: dict, path_to_csv: str):
    # first write keys then values
    with open(path_to_csv, 'w') as f:
        f.write(','.join(dictionary.keys()) + '\n')
        f.write(','.join([str(x) for x in dictionary.values()]) + '\n')
