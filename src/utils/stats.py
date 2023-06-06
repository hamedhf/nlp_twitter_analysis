from hazm import lemmatizer


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
        segments = line.split(',')
        for segment in segments:
            if segment != 'PAD':
                count += 1

    return count


def get_unique_word_count(path_to_word_csv: str):
    with open(path_to_word_csv, 'r') as f:
        lines = f.readlines()

    words = set()
    lm = lemmatizer.Lemmatizer()
    for line in lines:
        segments = line.split(',')
        for segment in segments:
            if segment != 'PAD':
                words.add(lm.lemmatize(segment))

    return len(words)


def write_dict_to_csv(dictionary: dict, path_to_csv: str):
    # first write keys then values
    with open(path_to_csv, 'w') as f:
        f.write(','.join(dictionary.keys()) + '\n')
        f.write(','.join([str(x) for x in dictionary.values()]) + '\n')
