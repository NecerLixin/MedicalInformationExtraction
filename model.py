import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import GCNConv, GATConv
class NerModelLSTM(nn.Module):
    def __init__(self, num_embeddings, embedding_dim,num_labels,hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=embedding_dim,
                                      )
        self.crf_layer = CRFLayer(num_labels)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=2,bidirectional=True,
                            )
        self.fc = nn.Linear(2*hidden_size, num_labels)
        # [batch,len,emb]
        # [len,batch,emb]
    def forward(self,train_x):
        train_x = self.embedding(train_x)
        train_x = train_x.permute([1,0,2])
        out,_ = self.lstm(train_x)
        out = self.fc(out).permute([1,0,2])
        return out
    def decode(self,emission):
        tag = self.crf_layer.decode(emission)
        return tag

class NerModelBert(nn.Module):
    def __init__(self, bert_model:BertModel,num_labels,hidden_size=768) -> None:
        super().__init__()
        self.bert_model = bert_model
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.crf_layer = CRFLayer(num_labels)
        self.cls = nn.Linear(self.hidden_size,num_labels)

    def forward(self,inputs):
        x = self.bert_model(**inputs).last_hidden_state
        emission = self.cls(x)
        # tag = self.crf_layer.decode(tag,tag.shape[1])
        return emission
    
    def decode(self,emission):
        tag = self.crf_layer.decode(emission)
        return tag
        
        
class CRFLayer(nn.Module):
    def __init__(self, num_tags):
        super(CRFLayer,self).__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags,num_tags))
        self.crf = CRF(num_tags=self.num_tags,batch_first=True)
        self.crf.transitions = self.transitions
        
        
    def forward(self, inputs, sequence_lengths):
        # inputs: [batch_size, seq_len, num_features]
        # sequence_lengths: [batch_size]
        return inputs, sequence_lengths
    
    def decode(self,inputs,lengths=None):
        return self.crf.decode(inputs)

class ClsModelBertBase(nn.Module):
    def __init__(self,bert_model,hidden_size=768,num_symptoms=331,num_labels=3) -> None:
        super().__init__()
        self.bert_model = bert_model
        self.hidden_size = hidden_size
        self.cls = nn.Linear(hidden_size,num_labels)
        self.symptom_matrix = nn.Parameter(torch.randn(num_symptoms,hidden_size))
        
    def forward(self,input_ids,attention_mask,):
        embedding = self.bert_model(input_ids=input_ids,
                            attention_mask=attention_mask).last_hidden_state
        # [batch, seq_len, emb]
        embedding = embedding[:,0,:]#[batch,emb] 使用cls的 embedding 代表作为句子的 embedding
        embedding = embedding.unsqueeze(dim=1) # [b,1,e]
        embedding = torch.tanh(embedding * self.symptom_matrix.unsqueeze(dim=0)) # [b,1,e] * [1,m,e] = [b,m,e]
        output = self.cls(embedding) # [batch,num_labels]
        return output

class ClsModelBertSyntaxTree(nn.Module):
    def __init__(self,bert_model,hidden_size=768,num_symptoms=331,num_labels=3,gat_heads=4) -> None:
        super().__init__()
        self.bert_model = bert_model
        self.hidden_size = hidden_size
        self.gat_heads = gat_heads
        self.trans = nn.Linear(hidden_size*gat_heads,hidden_size)
        self.cls = nn.Linear(hidden_size,num_labels)
        self.symptom_matrix = nn.Parameter(torch.randn(num_symptoms,hidden_size))
        self.gat1 = GATConv(hidden_size,hidden_size,heads=gat_heads)

    def forward(self, input_ids, attention_mask, edge_index_list:list):
        embedding = self.bert_model(input_ids=input_ids,
                            attention_mask=attention_mask).last_hidden_state
        batch_size = len(edge_index_list)
        graph_data_list = []
        for i in range(batch_size):
            edge_index = torch.tensor(edge_index_list[i]).to(input_ids.device).T
            data = Data(x=embedding[i],edge_index=edge_index)
            graph_data_list.append(data)
        batch_data = Batch.from_data_list(graph_data_list)
        split_sizes = batch_data.ptr[1:] - batch_data.ptr[:-1]
        embedding = self.gat1(batch_data.x,batch_data.edge_index)
        
        embedding = embedding.split(split_sizes.tolist()) # [b,l,e]
        embedding = torch.stack(embedding,dim=0)
        embedding = self.trans(embedding)
        embedding = embedding[:,0,:]#[batch,emb] 使用cls的 embedding 代表作为句子的 embedding
        embedding = embedding.unsqueeze(dim=1) # [b,1,e]
        embedding = torch.tanh(embedding * self.symptom_matrix.unsqueeze(dim=0)) # [b,1,e] * [1,m,e] = [b,m,e]
        output = self.cls(embedding) # [batch,num_labels]
        return output


        
        