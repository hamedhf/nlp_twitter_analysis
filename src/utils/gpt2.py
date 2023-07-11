import datetime
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wget
from torch.optim import AdamW
from torch.utils.data import Dataset, random_split, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, GPT2LMHeadModel, GPT2Config, get_linear_schedule_with_warmup

from .constants import TOPICS

MODEL_NAME = 'HooshvareLab/gpt2-fa'
CACHE_DIR = '../models/huggingface_cache'
BOS_TOKEN = '<s>'
START_OF_TEXT_TOKEN = '<|startoftext|>'  # this token is used to indicate start of text for language modeling task
EOS_TOKEN = '</s>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'


def save_list_to_file(csv_path: str, l: list[str]):
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('tweet\n')
        for item in l:
            f.write(item + '\n')


def prepare_language_model_dataset(path_to_augmented_csv: str):
    with open(path_to_augmented_csv, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]  # skip header

    data = {label: [] for label in TOPICS.values()}

    for line in lines:
        _, _, tweet, _, _, label = line.split(',')
        if tweet == '' or tweet is None:
            continue
        label = label[:-1]  # remove \n
        data[label].append(tweet)

    base_path = '../data/languagemodel'
    for label in data.keys():
        random.shuffle(data[label])
        save_path = os.path.join(base_path, label + '.csv')
        save_list_to_file(save_path, data[label])


class TweetDataset(Dataset):
    def __init__(self, tweets, tokenizer, max_length=128):
        self.tokenizer = tokenizer  # the gp2 tokenizer we instantiated
        self.input_ids = []
        self.attn_masks = []

        for tweet in tweets:
            """
            This loop will iterate through each entry in tweets and tokenize it.
            For each bit of text it will prepend it with the start of text token,
            then append the end of text token and pad to the maximum length with the 
            pad token. 
            """
            txt = BOS_TOKEN + tweet + EOS_TOKEN
            encodings_dict = tokenizer(txt, truncation=True, max_length=max_length, padding="max_length")

            """
            Each iteration then appends either the encoded tensor to a list,
            or the attention mask for that encoding to a list. The attention mask is
            a binary list of 1's or 0's which determine whether the langauge model
            should take that token into consideration or not. 
            """
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round(elapsed))))


def train_op(model, train_dataloader, validation_dataloader, epochs, device, optimizer, scheduler):
    total_t0 = time.time()
    training_stats = []
    model = model.to(device)

    for epoch_i in tqdm(range(0, epochs), position=0):
        print(f'Beginning epoch {epoch_i + 1} of {epochs}')
        t0 = time.time()
        total_train_loss = 0
        model.train()

        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), position=0):
            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)
            model.zero_grad()
            outputs = model(b_input_ids, labels=b_labels, attention_mask=b_masks, token_type_ids=None)
            loss = outputs[0]
            batch_loss = loss.item()
            total_train_loss += batch_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Calculate the average loss over all the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print()
        print(f'Average Training Loss: {avg_train_loss}. Epoch time: {training_time}')
        print()

        t0 = time.time()
        model.eval()
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in tqdm(validation_dataloader, total=len(validation_dataloader), position=0):
            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)

            with torch.no_grad():
                outputs = model(b_input_ids, attention_mask=b_masks, labels=b_labels)
                loss = outputs[0]

            batch_loss = loss.item()
            total_eval_loss += batch_loss

        avg_val_loss = total_eval_loss / len(validation_dataloader)
        validation_time = format_time(time.time() - t0)

        print()
        print(f'Validation loss: {avg_val_loss}. Validation Time: {validation_time}')
        print()

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print(f'Total training took {format_time(time.time() - total_t0)}')
    return training_stats


def train_gpt2(label: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        unk_token=UNK_TOKEN,
        cache_dir=CACHE_DIR
    )
    tokenizer.add_special_tokens({
        "bos_token": BOS_TOKEN,
        "eos_token": EOS_TOKEN,
        "pad_token": PAD_TOKEN,
        "unk_token": UNK_TOKEN
    })

    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        bos_token_id=tokenizer(BOS_TOKEN)["input_ids"][0],
        eos_token_id=tokenizer(EOS_TOKEN)["input_ids"][0],
        pad_token_id=tokenizer(PAD_TOKEN)["input_ids"][0],
        unk_token_id=tokenizer(UNK_TOKEN)["input_ids"][0],
        cache_dir=CACHE_DIR
    )

    save_path = f'../models/gpt2/{label}/'
    tokenizer.save_pretrained(save_path)
    config.save_pretrained(save_path)

    model_url = "https://huggingface.co/HooshvareLab/gpt2-fa/resolve/main/pytorch_model.bin"
    tokenizer_url = "https://huggingface.co/HooshvareLab/gpt2-fa/resolve/main/tokenizer.json"

    # Download the model
    base_model_path = '../models/gpt2/base_persian/pytorch_model.bin'
    if not os.path.exists(base_model_path):
        wget.download(model_url, base_model_path)
        print("\nModel downloaded successfully!")
    else:
        print("Model already exists!")

    # copy the model to the destination and if exists, overwrite it
    model_destination = "../models/gpt2/" + label + "/pytorch_model.bin"
    os.system(f'cp {base_model_path} {model_destination}')
    print("Model copied successfully!")

    # Download the tokenizer
    base_tokenizer_path = '../models/gpt2/base_persian/tokenizer.json'
    if not os.path.exists(base_tokenizer_path):
        wget.download(tokenizer_url, base_tokenizer_path)
        print("\ntokenizer.json downloaded successfully!")
    else:
        print("tokenizer.json already exists!")

    # copy the tokenizer to the destination and if exists, overwrite it
    tokenizer_destination = "../models/gpt2/" + label + "/tokenizer.json"
    os.system(f'cp {base_tokenizer_path} {tokenizer_destination}')
    print("tokenizer.json copied successfully!")

    tokenizer = AutoTokenizer.from_pretrained(
        save_path,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN
    )

    csv_path = f'../data/languagemodel/{label}.csv'
    tweets = pd.read_csv(csv_path)['tweet'].tolist()

    # max_seq = max([len(tokenizer.encode(tweet)) for tweet in tweets])
    # Due to the limited resources and for the sake of simplicity and speed, we set the max_seq to 128
    max_seq = 128

    dataset = TweetDataset(tweets, tokenizer, max_length=max_seq)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=8
    )

    validation_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=8
    )

    # Loading the model configuration and setting it to the GPT2 standard settings.
    configuration = GPT2Config.from_pretrained(save_path, output_hidden_states=False)

    # Create the instance of the model and set the token size embedding length
    model = GPT2LMHeadModel.from_pretrained(save_path, config=configuration)
    model.resize_token_embeddings(len(tokenizer))

    # Tell pytorch to run this model on the GPU.
    model = model.to(device)

    # This step is optional but will enable reproducible runs.
    # seed_val = 42
    # random.seed(seed_val)
    # np.random.seed(seed_val)
    # torch.manual_seed(seed_val)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed_val)

    epochs = 4
    warmup_steps = 1e2

    optimizer = AdamW(
        model.parameters(),
        lr=5e-4,
        eps=1e-8
    )

    """
    Total training steps is the number of data points, times the number of epochs. 
    Essentially, epochs are training cycles, how many times each point will be seen by the model. 
    """
    total_steps = len(train_dataloader) * epochs

    """
    We can set a variable learning rate which will help scan larger areas of the 
    problem space at higher LR earlier, then fine tune to find the exact model minima 
    at lower LR later in training.
    """
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    training_stats = train_op(model, train_dataloader, validation_dataloader, epochs, device, optimizer, scheduler)
    path_to_plot = f'../stats/plot_gpt2_finetuning_{label}.png'
    pd.options.display.max_rows = 5
    # pd.set_option('precision', 2)
    df_stats = pd.DataFrame(data=training_stats)
    df_stats = df_stats.set_index('epoch')

    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)

    # Plot the learning curve.
    plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
    plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

    # Label the plot.
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks([1, 2, 3, 4])
    plt.savefig(path_to_plot)
    # clear plt and sns memory
    sns.reset_orig()
    plt.close("all")
    plt.clf()

    # Saving the model and the tokenizer
    path_to_model = f'../models/gpt2/{label}/'
    model.save_pretrained(path_to_model)
    tokenizer.save_pretrained(path_to_model)
    configuration.save_pretrained(path_to_model)


def gpt2_generator(prompt: str, label: str, device: torch.device, max_length=128, num_return_sequences=3) -> list:
    model_folder_path = f'../models/gpt2/{label}/'
    model_path = model_folder_path + 'pytorch_model.bin'
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model doesn't exist, please train it first!")

    configuration = GPT2Config.from_pretrained(model_folder_path, output_hidden_states=False)
    model = GPT2LMHeadModel.from_pretrained(model_folder_path, config=configuration)
    tokenizer = AutoTokenizer.from_pretrained(
        model_folder_path,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN
    )
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    model.eval()

    prompt = BOS_TOKEN + prompt + START_OF_TEXT_TOKEN
    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    generated = generated.to(device)
    print(f"This is what the model is given as input: {prompt}")

    decoded_outputs = model.generate(
        generated,
        do_sample=True,
        top_k=50,
        max_length=max_length,
        top_p=0.95,
        num_return_sequences=num_return_sequences
    )
    outputs = []
    for i, output in enumerate(decoded_outputs):
        gen_sample_output = tokenizer.decode(output, skip_special_tokens=True)
        outputs.append(gen_sample_output)
    return outputs
