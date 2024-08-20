from collections import Counter
import os
import requests
import tarfile
from typing import List
import nltk
import nltk.tokenize
import torch
import tqdm


class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, lang, max_len=100):

        # encode the text
        encoded_texts = [
            [lang.word2index.get(word, 0) for word in nltk.tokenize.word_tokenize(text)]
            for text in tqdm.tqdm(texts, desc="Encoding text")
        ]

        # align the length of the text
        self.max_len = max_len
        self.texts = [
            text[:max_len] + [1] * (max_len - len(text))
            for text in tqdm.tqdm(encoded_texts, desc="Padding text")
        ]
        self.labels = labels
        self.lang = lang

    def __getitem__(self, idx):
        return (
            torch.tensor(self.texts[idx]),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )

    def __len__(self):
        return len(self.texts)


class Lang:
    def __init__(self):
        self.word2index = {}
        self.index2word = {0: "<unk>", 1: "<pad>"}
        self.n_words = 2
        self.word_counter = Counter()

    def add_text(self, text: List[str]):
        for sentence in tqdm.tqdm(text, desc="Building vocabulary"):
            self.word_counter.update(nltk.tokenize.word_tokenize(sentence))

        top_words = self.word_counter.most_common()

        # encode
        self.word2index = {word: i + 2 for i, (word, _) in enumerate(top_words)}
        self.word2index.update({"<unk>": 0, "<pad>": 1})
        # decode
        self.index2word = {i: word for word, i in self.word2index.items()}

        self.n_words = len(self.word2index)


def download_imdb(data_dir):
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    response = requests.get(url, stream=True)
    with open(os.path.join(data_dir, "aclImdb_v1.tar.gz"), "wb") as f:
        f.write(response.content)

    with tarfile.open(os.path.join(data_dir, "aclImdb_v1.tar.gz"), "r:gz") as f:
        f.extractall(data_dir)


def read_imdb(data_dir, limit: int = None):
    if not os.path.exists(data_dir):
        download_imdb(data_dir)

    lang = Lang()

    train_texts = []
    train_labels = []

    for label in ["pos", "neg"]:
        for file in tqdm.tqdm(
            os.listdir(os.path.join(data_dir, "aclImdb", "train", label))[:limit],
            desc=f"Reading {label} train files",
        ):
            with open(
                os.path.join(data_dir, "aclImdb", "train", label, file),
                "r",
                encoding="utf-8",
            ) as f:
                train_texts.append(f.read())
                train_labels.append(0 if label == "neg" else 1)

    # only add train texts to the lang
    lang.add_text(train_texts)

    test_texts = []
    test_labels = []
    for label in ["pos", "neg"]:
        for file in tqdm.tqdm(
            os.listdir(os.path.join(data_dir, "aclImdb", "test", label))[:limit],
            desc=f"Reading {label} test files",
        ):
            with open(
                os.path.join(data_dir, "aclImdb", "test", label, file),
                "r",
                encoding="utf-8",
            ) as f:
                test_texts.append(f.read())
                test_labels.append(0 if label == "neg" else 1)

    train_dataset = IMDBDataset(train_texts, train_labels, lang)
    test_dataset = IMDBDataset(test_texts, test_labels, lang)
    return train_dataset, test_dataset, lang


if __name__ == "__main__":
    _ = read_imdb("data")
