import os
import random

from transformers import AutoConfig, AutoTokenizer, AutoModel

from .constants import TOPICS


def tuplize(tweets: list[str], label: str) -> list[tuple[str, str]]:
    return [(tweet, label) for tweet in tweets]


def save_list_to_file(csv_name: str, l: list[tuple[str, str]]):
    csv_path = os.path.join('../data/parsbert', csv_name)
    with open(csv_path, 'w', encoding='utf-8') as f:
        for tweet, label in l:
            f.write(f"{tweet},{label}\n")


def prepare_dataset(path_to_csv: str):
    train, validation, test = [], [], []
    with open(path_to_csv, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]  # skip header

    data = {}
    for label in TOPICS.values():
        data[label] = []

    for line in lines:
        _, _, tweet, _, _, label = line.split(',')
        label = label[:-1]  # remove \n
        data[label].append(tweet)

    # split data into train, validation, test
    for label in TOPICS.values():
        count = len(data[label])
        train_count = int(count * 0.5)
        validation_count = int(count * 0.25)
        test_count = count - train_count - validation_count
        print(f"splitted {label} into {train_count} train, {validation_count} validation, {test_count} test")

        # shuffle data
        random.shuffle(data[label])
        train.extend(tuplize(data[label][:train_count], label))
        validation.extend(tuplize(data[label][train_count:train_count + validation_count], label))
        test.extend(tuplize(data[label][train_count + validation_count:], label))

    random.shuffle(train)
    random.shuffle(validation)
    random.shuffle(test)
    save_list_to_file('train.csv', train)
    save_list_to_file('validation.csv', validation)
    save_list_to_file('test.csv', test)


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cache_dir = base_dir + "/models/huggingface_cache"
    config = AutoConfig.from_pretrained("HooshvareLab/bert-base-parsbert-uncased", cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased", cache_dir=cache_dir)
    model = AutoModel.from_pretrained("HooshvareLab/bert-base-parsbert-uncased", cache_dir=cache_dir)
    text = "ما در هوشواره معتقدیم با انتقال صحیح دانش و آگاهی، همه افراد می‌توانند از ابزارهای هوشمند استفاده کنند. شعار ما هوش مصنوعی برای همه است."  # noqa
    print(tokenizer.tokenize(text))
