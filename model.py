import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import GCNConv, GATConv


class NerModelLSTM(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_labels, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )
        self.crf_layer = CRFLayer(num_labels)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
        )
        self.fc = nn.Linear(2 * hidden_size, num_labels)
        # [batch,len,emb]
        # [len,batch,emb]

    def forward(self, train_x):
        train_x = self.embedding(train_x)
        train_x = train_x.permute([1, 0, 2])
        out, _ = self.lstm(train_x)
        out = self.fc(out).permute([1, 0, 2])
        return out

    def decode(self, emission):
        tag = self.crf_layer.decode(emission)
        return tag


class NerModelLSTMWord(nn.Module):
    def __init__(
        self, num_characters, num_words, embedding_dim, num_labels, hidden_size
    ):
        super().__init__()
        self.num_characters = num_characters
        self.num_words = num_words
        self.hidden_size = embedding_dim
        self.embedding_char = nn.Embedding(
            num_embeddings=num_characters,
            embedding_dim=embedding_dim,
        )
        self.embedding_word = nn.Embedding(
            num_embeddings=num_words,
            embedding_dim=embedding_dim,
        )
        self.crf_layer = CRFLayer(num_labels)
        self.lstm_char = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
        )
        self.lstm_word = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
        )
        self.q = nn.Linear(2 * hidden_size, hidden_size)
        self.k = nn.Linear(2 * hidden_size, hidden_size)
        self.v = nn.Linear(2 * hidden_size, 2 * hidden_size)

        self.fc = nn.Linear(2 * hidden_size, num_labels)
        # [batch,len,emb]
        # [len,batch,emb]

    def forward(self, x_char, x_word):
        x_char = self.embedding_char(x_char).permute([1, 0, 2])
        x_word = self.embedding_word(x_word).permute([1, 0, 2])
        out_char, _ = self.lstm_char(x_char)  # [b, l, e]
        out_word, _ = self.lstm_word(x_word)  # [b, s, e]
        out_char = out_char.permute([1, 0, 2])
        out_word = out_word.permute([1, 0, 2])
        q = self.q(out_word)  # [b,l,e/2]
        k = self.k(out_char)  # [b,s,e/2]
        v = self.v(out_word)  # [b,l,e]
        scale = torch.sqrt(torch.tensor(self.hidden_size * 2)).to(q.device)
        attention = torch.einsum("ble,bse->bsl", q, k) / scale  # [b,s,l]
        residual = torch.einsum("bsl,ble->bse", attention, v)  # [b,s,e]
        out_char = out_char + residual
        out_char = self.fc(out_char)
        return out_char

    def decode(self, emission):
        tag = self.crf_layer.decode(emission)
        return tag


class NerModelBert(nn.Module):
    def __init__(self, bert_model: BertModel, num_labels, hidden_size=768) -> None:
        super().__init__()
        self.bert_model = bert_model
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.crf_layer = CRFLayer(num_labels)
        self.cls = nn.Linear(self.hidden_size, num_labels)

    def forward(self, inputs):
        x = self.bert_model(**inputs).last_hidden_state
        emission = self.cls(x)
        # tag = self.crf_layer.decode(tag,tag.shape[1])
        return emission

    def decode(self, emission):
        tag = self.crf_layer.decode(emission)
        return tag


class CRFLayer(nn.Module):
    def __init__(self, num_tags):
        super(CRFLayer, self).__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.crf = CRF(num_tags=self.num_tags, batch_first=True)
        self.crf.transitions = self.transitions

    def forward(self, inputs, sequence_lengths):
        # inputs: [batch_size, seq_len, num_features]
        # sequence_lengths: [batch_size]
        return inputs, sequence_lengths

    def decode(self, inputs, lengths=None):
        return self.crf.decode(inputs)


class ClsModelBertBase(nn.Module):
    def __init__(
        self, bert_model, hidden_size=768, num_symptoms=331, num_labels=3
    ) -> None:
        super().__init__()
        self.bert_model = bert_model
        self.hidden_size = hidden_size
        self.cls = nn.Linear(hidden_size, num_labels)
        self.symptom_matrix = nn.Parameter(torch.randn(num_symptoms, hidden_size))

    def forward(
        self,
        input_ids,
        attention_mask,
    ):
        embedding = self.bert_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        # [batch, seq_len, emb]
        embedding = embedding[
            :, 0, :
        ]  # [batch,emb] 使用cls的 embedding 代表作为句子的 embedding
        embedding = embedding.unsqueeze(dim=1)  # [b,1,e]
        embedding = torch.tanh(
            embedding * self.symptom_matrix.unsqueeze(dim=0)
        )  # [b,1,e] * [1,m,e] = [b,m,e]
        output = self.cls(embedding)  # [batch,num_labels]
        return output


class ClsModelBertBase2(nn.Module):
    def __init__(
        self, bert_model, hidden_size=768, num_symptoms=331, num_labels=3
    ) -> None:
        super().__init__()
        self.bert_model = bert_model
        self.hidden_size = hidden_size
        self.cls = nn.Linear(hidden_size, num_labels)
        self.symptom_matrix = nn.Parameter(torch.randn(num_symptoms, hidden_size))

    def forward(
        self,
        input_ids,
        attention_mask,
    ):
        embedding = self.bert_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        # [batch, seq_len, emb]
        embedding = embedding[:, 0, :]
        # [batch,emb] 使用cls的 embedding 代表作为句子的 embedding
        embedding = embedding.unsqueeze(dim=1)  # [b,1,e]
        embedding = torch.tanh(
            embedding * self.symptom_matrix.unsqueeze(dim=0)
        )  # [b,1,e] * [1,m,e] = [b,m,e]
        output = self.cls(embedding)  # [batch,num_labels]
        return output


class ClsModelBertSyntaxTree(nn.Module):
    def __init__(
        self, bert_model, hidden_size=768, num_symptoms=331, num_labels=3, gat_heads=4
    ) -> None:
        super().__init__()
        self.bert_model = bert_model
        self.hidden_size = hidden_size
        self.gat_heads = gat_heads
        self.trans = nn.Linear(hidden_size * gat_heads, hidden_size)
        self.cls = nn.Linear(hidden_size, num_labels)
        self.symptom_matrix = nn.Parameter(torch.randn(num_symptoms, hidden_size))
        self.gat1 = GATConv(hidden_size, hidden_size, heads=gat_heads)

    def forward(self, input_ids, attention_mask, edge_index_list: list):
        embedding = self.bert_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        batch_size = len(edge_index_list)
        graph_data_list = []
        for i in range(batch_size):
            edge_index = torch.tensor(edge_index_list[i]).to(input_ids.device).T
            data = Data(x=embedding[i], edge_index=edge_index)
            graph_data_list.append(data)
        batch_data = Batch.from_data_list(graph_data_list)
        split_sizes = batch_data.ptr[1:] - batch_data.ptr[:-1]
        embedding = self.gat1(batch_data.x, batch_data.edge_index)

        embedding = embedding.split(split_sizes.tolist())  # [b,l,e]
        embedding = torch.stack(embedding, dim=0)
        embedding = self.trans(embedding)
        embedding = embedding[:, 0, :]
        # [batch,emb] 使用cls的 embedding 代表作为句子的 embedding
        embedding = embedding.unsqueeze(dim=1)  # [b,1,e]
        embedding = torch.tanh(
            embedding * self.symptom_matrix.unsqueeze(dim=0)
        )  # [b,1,e] * [1,m,e] = [b,m,e]
        output = self.cls(embedding)  # [batch,num_labels]
        return output


class ClsModelLSTM(nn.Module):
    def __init__(
        self, num_embeddings, embedding_dim, num_labels, hidden_size, num_symptoms
    ):
        super().__init__()
        self.num_symptoms = num_symptoms
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
        )
        self.fc = nn.Linear(2 * hidden_size, num_labels)
        self.symptom_matrix = nn.Parameter(torch.randn(num_symptoms, hidden_size * 2))

        # [batch,len,emb]
        # [len,batch,emb]

    def forward(self, input_ids):
        embedding = self.embedding(input_ids)
        embedding = embedding.permute([1, 0, 2])
        embedding, _ = self.lstm(embedding)
        embedding = embedding.permute([1, 0, 2])
        embedding = embedding[:, 0, :]
        embedding = embedding.unsqueeze(dim=1)  # [b,1,e]
        embedding = torch.tanh(embedding * self.symptom_matrix.unsqueeze(dim=0))
        out = self.fc(embedding)
        return out


class NormModelBERT(nn.Module):
    def __init__(
        self, bert_model: BertModel, num_labels, hidden_size=768, offset=1
    ) -> None:
        super().__init__()
        self.bert_model = bert_model
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.cls = nn.Linear(hidden_size, num_labels)
        self.offset = offset

    def forward(self, input_ids, attention_mask, entity_pos: list):
        embedding = self.bert_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        batch = input_ids.shape[0]
        entity_embeddings = []
        for i in range(batch):
            entity_pos_i = entity_pos[i]
            if len(entity_pos_i) > 0:
                entity_pos_i = (
                    torch.tensor(entity_pos[i]).to(input_ids.device) + self.offset
                )
                entity_pos_i = entity_pos_i[:, 0]
                entity_embedding = torch.index_select(
                    embedding[i], dim=0, index=entity_pos_i
                )  # [m, e]
                entity_embeddings.append(entity_embedding)
        entity_embedding = torch.cat(entity_embeddings, dim=0)  # [n,e]
        output = self.cls(entity_embedding)  # [n,3]
        return output
