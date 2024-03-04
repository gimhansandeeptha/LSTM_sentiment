import re 
from bs4 import BeautifulSoup
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torchtext.data import get_tokenizer

import numpy as np
from embedding import GloveEmbedding


def clean_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, 'html.parser').get_text(" ")

    # Remove non-word characters and digits
    text = re.sub(r"[^\w\s.,]", '', text)
    text = re.sub(r"\d", '', text)

    # Convert to lowercase
    cleaned_text = text.lower()
    return cleaned_text



class CustomDataset(Dataset):
    def __init__(self, reviews, sentiments, word2idx, max_seq_length=500,tokenizer = None):
        self.reviews = reviews
        self.sentiments = sentiments
        self.word2idx = word2idx
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, index):
        review = self.reviews[index]
        label = self.sentiments[index]

        # Tokenize and convert words to indices
        tokens = self.tokenizer(review)
        indices = [self.word2idx.get(token, 0) for token in tokens][:self.max_seq_length]

        # Pad or truncate to a fixed sequence length
        pad_length = self.max_seq_length - len(indices)
        indices += [0] * pad_length  # 0 is used for padding

        # Convert indices to integers
        indices = [int(idx) for idx in indices]

        # One hot encoding
        one_hot_label = F.one_hot(torch.tensor(label), num_classes=2)

        return torch.tensor(indices), one_hot_label.float()

class Preprocess():
    def __inti__(self):
        pass
    def get_loader(text):
        # text = "This movie is great with lot of interesting scenes. really enjoyed the whole film. thank you"
        cleaned_text = clean_text(text)

        tokenizer = get_tokenizer('basic_english')

        glove_embedding = GloveEmbedding()

        vectors, words, word2idx, glove = glove_embedding.get_embedding()
        # DataLoader for training data
        x_train=np.asarray([cleaned_text])
        y_train =np.asarray([1])
        train_dataset = CustomDataset(x_train, y_train, word2idx, max_seq_length=500, tokenizer=tokenizer)
        train_loader = DataLoader(train_dataset)

        return train_loader