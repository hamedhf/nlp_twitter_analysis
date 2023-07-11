import io
import os
import random

import sentencepiece as spm

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


def train_spm_tokenizer(augmented_csv_path: str, vocab_size: int = 1500) -> int:
    if not augmented_csv_path.endswith('.csv') or not os.path.exists(augmented_csv_path):
        raise FileNotFoundError(f"{augmented_csv_path} not found. Please run augment script first.")

    corpus = []
    with open(augmented_csv_path, encoding='utf-8') as f:
        lines = f.readlines()[1:]  # skip header
        for line in lines:
            tweet = line.split(',')[2]
            corpus.append(tweet)

    random.shuffle(corpus)

    # 4/5 of the data is used for training
    train = corpus[:int(len(corpus) * 4 / 5)]
    test = corpus[int(len(corpus) * 4 / 5):]

    model = io.BytesIO()
    spm.SentencePieceTrainer.Train(
        sentence_iterator=iter(train),
        model_writer=model,
        model_type='unigram',  # 'bpe', 'char', 'word', 'unigram' (default)
        vocab_size=vocab_size,
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID
    )

    save_path = f'../models/spm_tokenizer/tokenizer{vocab_size}.model'
    with open(save_path, 'wb') as f:
        f.write(model.getvalue())
    print(f"Saved tokenizer to {save_path}")

    # count unk tokens in test set
    sp = spm.SentencePieceProcessor(model_proto=model.getvalue())
    unk_count = 0
    for tweet in test:
        encoded_input = sp.Encode(tweet)
        for token_id in encoded_input:
            if token_id == UNK_ID:
                unk_count += 1

    return unk_count
