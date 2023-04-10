import torch
from torch import nn, optim
from transformers import AutoModel, AutoTokenizer


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
        x = x.last_hidden_state[:, 0] # [cls] emb
        x = self.classifier(x)
        return x