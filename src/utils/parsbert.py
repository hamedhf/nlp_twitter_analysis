import os

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .constants import TOPICS
from .stats import write_dict_to_csv


class TweetDataset(Dataset):
    def __init__(self, tweets, labels, tokenizer):
        self.tweets = tweets
        self.labels = labels
        self.tokenizer = tokenizer
        self.label2idx = {label: idx for idx, label in enumerate(TOPICS.values())}

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        tweet = str(self.tweets[idx])
        label = self.label2idx[self.labels[idx]]

        encoding = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long),
        }


def get_tweet_and_label_from_csv(csv_path: str) -> tuple[list[str], list[str]]:
    df = pd.read_csv(csv_path)
    tweets = df['tweet'].tolist()
    labels = df['label'].tolist()
    return tweets, labels


def train_parsbert(device: torch.device):
    # parsbert v1, link: https://huggingface.co/HooshvareLab/bert-base-parsbert-uncased
    # parsbert v2, link: https://huggingface.co/HooshvareLab/bert-fa-base-uncased
    model_name = 'HooshvareLab/bert-fa-base-uncased'  # parsbert v2
    cache_dir = '../models/huggingface_cache'
    train_csv_path = '../data/split/train.csv'
    validation_csv_path = '../data/split/validation.csv'

    train_tweets, train_labels = get_tweet_and_label_from_csv(train_csv_path)
    val_tweets, val_labels = get_tweet_and_label_from_csv(validation_csv_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    train_dataset = TweetDataset(train_tweets, train_labels, tokenizer)
    val_dataset = TweetDataset(val_tweets, val_labels, tokenizer)

    batch_size = 16
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(TOPICS.values()),
                                                               cache_dir=cache_dir)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    epochs = 10
    early_stop_patience = 3
    best_val_loss = float('inf')
    best_epoch = 0
    no_improvement_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        description = f"Training Epoch {epoch + 1}"
        progress_bar = tqdm(train_dataloader, desc=description, colour='green')
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)  # noqa
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.logits, dim=1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()  # noqa
            progress_bar.set_postfix({"Loss": loss.item()})

        train_average_loss = train_loss / len(train_dataloader)
        train_accuracy = train_correct / train_total
        print(f"Train Loss: {train_average_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")

        # Save the fine-tuned model
        output_dir = "../models/parsbert/final"
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Evaluate the model
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            description = f"Validation Epoch {epoch + 1}"
            progress_bar = tqdm(val_dataloader, desc=description, colour='yellow')
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(device)  # noqa
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = outputs.loss
                val_loss += loss.item()

                _, predicted = torch.max(outputs.logits, dim=1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()  # noqa

            val_average_loss = val_loss / len(val_dataloader)
            val_accuracy = val_correct / val_total
            print(f"Validation Loss: {val_average_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improvement_counter = 0

            # Save the best fine-tuned model
            output_dir = "../models/parsbert/best"
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        else:
            no_improvement_counter += 1
            if no_improvement_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1} to prevent overfitting. Best epoch: {best_epoch + 1}")
                break


def test_parsbert_model(device: torch.device):
    model_path_folder = "../models/parsbert/best"
    model_path = "../models/parsbert/best/pytorch_model.bin"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found, please train the model first and then test it.")

    tokenizer = AutoTokenizer.from_pretrained(model_path_folder)
    model = AutoModelForSequenceClassification.from_pretrained(model_path_folder, num_labels=len(TOPICS.values()))
    model = model.to(device)

    batch_size = 16
    test_csv_path = '../data/split/test.csv'
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(
            f"Test CSV file not found, please split the data first and then test the model.(run train script)")
    test_tweets, test_labels = get_tweet_and_label_from_csv(test_csv_path)
    test_dataset = TweetDataset(test_tweets, test_labels, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        description = f"Testing"
        progress_bar = tqdm(test_dataloader, desc=description, colour='yellow')
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            test_loss += loss.item()

            _, predicted = torch.max(outputs.logits, dim=1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()  # noqa

            progress_bar.set_postfix({"Loss": loss.item()})

        test_average_loss = test_loss / len(test_dataloader)
        test_accuracy = test_correct / test_total
        print(f"Test Loss: {test_average_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")

        # save the test results in a csv file
        test_average_loss = round(test_average_loss, 4)
        test_accuracy = round(test_accuracy, 4)
        test_results = {
            "test-loss": test_average_loss,
            "test-accuracy": test_accuracy
        }
        write_dict_to_csv(test_results, "../stats/parsbert_test_results.csv")


def classify_tweet(tweet: str, device: torch.device) -> str:
    model_path_folder = "../models/parsbert/best"
    model_path = "../models/parsbert/best/pytorch_model.bin"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found, please train the model first and use it for classification.")

    tokenizer = AutoTokenizer.from_pretrained(model_path_folder)
    model = AutoModelForSequenceClassification.from_pretrained(model_path_folder, num_labels=len(TOPICS.values()))
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        inputs = tokenizer(tweet, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        _, predicted = torch.max(outputs.logits, dim=1)
        predicted_label = predicted.item()
        predicted_label = TOPICS[predicted_label]
    return predicted_label


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cache_dir = base_dir + "/models/huggingface_cache"
    tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased", cache_dir=cache_dir)
    text = "ما در هوشواره معتقدیم با انتقال صحیح دانش و آگاهی، همه افراد می‌توانند از ابزارهای هوشمند استفاده کنند. شعار ما هوش مصنوعی برای همه است."  # noqa
    print(tokenizer.tokenize(text))


if __name__ == '__main__':
    main()
