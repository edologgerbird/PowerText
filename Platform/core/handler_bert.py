import pandas as pd
import numpy as np
from random import randint, shuffle
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DS(Dataset):
    def __init__(self, data_path, sheet_name, model_ckpt, max_token_length=50):
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
        # load excel
        self.data = pd.read_csv(data_path)
        # self.data = pd.read_csv(data_path)
        # self.data = data.reset_index()
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data.loc[idx, :]
        comment = row['body']
        comment = comment.split()
        # long comments
        if len(comment) > self.max_token_length:
            comment = comment[:45]
        comment = ' '.join(comment)
        comment = comment.replace("\\", "")
        emotion = row['label']
        emotion = self.labels.index(emotion)
        return f"{comment}", emotion
    def choose(self):
        return self[randint(0, len(self)-1)]
    def get_tokenizer_size(self):
        return len(self.tokenizer)
    def decode(self, input_id):
        return self.tokenizer.decode(input_id)
    def collate_fn(self, data):
        comments, emotions = zip(*data)
        comments = self.tokenizer(comments, padding=True, return_tensors='pt')
        comments = {k:v.to(device) for k, v in comments.items()}
        emotions = torch.tensor(emotions).long().to(device)
        return comments, emotions


class SingleClassifier(nn.Module):
    def __init__(self, model_ckpt, nlabels, tokenizer_size=30523):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_ckpt)
        self.encoder.resize_token_embeddings(tokenizer_size)
        encoder_config = self.encoder.config
        self.classifier = nn.Sequential(
            nn.LayerNorm(encoder_config.hidden_size),
            nn.Dropout(0.3),
            nn.Linear(encoder_config.hidden_size, nlabels)
        )
    def get_summary(self):
        print(self)
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
    def forward(self, x):
        x = self.encoder(**x)
        x = x.last_hidden_state[:, 0]  # [cls] emb
        x = self.classifier(x)
        return x


# create pretrained model
model_ckpt = "GroNLP/hateBERT"
batch_size = 32

model = SingleClassifier(model_ckpt, nlabels=8).to(device)
model.unfreeze_encoder()
model_weight = torch.load("final_model_weights_full.pth", map_location=torch.device('cpu'))
model.load_state_dict(model_weight)


def create_csv(text):
    data = [[text, "neutral"]]
    data = pd.DataFrame(data)
    data.columns = ['body', 'label']
    data.to_csv("eval.csv", index=False)


def run_hate_bert(text):
    # export text input to csv format
    create_csv(text)

    # prediction
    input_ds = DS("eval.csv", "Sheet1", model_ckpt)
    input_dl = DataLoader(input_ds, batch_size=batch_size, collate_fn=input_ds.collate_fn)

    with torch.no_grad():
        for i in input_dl:
            comments, label = i
            label_outputs = model(comments)

    return list(label_outputs.numpy()[0])
