
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
import torch
from random import randint
import streamlit as st

class DS(Dataset):
    def __init__(self, data, model_ckpt, max_token_length=50):
        super().__init__()
        self.max_token_length = max_token_length

        # label encodings
        self.labels = [
            'hate',
            'privacy',
            'sexual',
            'impersonation',
            'illegal',
            'advertisement',
            'ai',
            'neutral'
        ]

        # load data
        self.data = data.reset_index(drop=True)
        self.data['label'] = 0

        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx, :]


        comment = row['body']
        comment = str(comment)
        comment = comment.split()

        # long comments
        if len(comment) > self.max_token_length:
            comment = comment[:45]
        comment = ' '.join(comment)
        comment = comment.replace("\\", "")

        emotion = 0

        return f"{comment}", emotion

    def choose(self):
        return self[randint(0, len(self)-1)]

    def get_tokenizer_size(self):
        return len(self.tokenizer)

    def decode(self, input_id):
        return self.tokenizer.decode(input_id)

    def collate_fn(self, data):
        comments, emotions = zip(*data)
        comments = self.tokenizer(comments,
                                  padding=True,
                                  return_tensors='pt')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        comments = {k:v.to(device) for k, v in comments.items()}
        emotions = torch.tensor(emotions).long().to(device)
        return comments, emotions