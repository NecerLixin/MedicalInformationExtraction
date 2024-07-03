import torch
import torch.nn as nn


class ModelLSTM(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=100,
            num_layers=2,
            bidirectional=True,
        )
        # [batch,len,emb]
        # [len,batch,emb]

    def forward(self, x):
        x = self.embedding(x)
        h, _ = self.lstm(x)
