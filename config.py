import torch


class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
