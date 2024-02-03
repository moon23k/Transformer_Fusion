import torch, copy, math
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel




def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def load_ple(config):
    ple = AutoModel.from_pretrained(config.mname)

    #Extend Max Position Embeddings
    if config.task == 'summarization':
        max_len = config.max_len
        embeddings = ple.embeddings
        ple_max_len = ple.config.max_position_embeddings

        temp_emb = nn.Embedding(max_len, ple.config.embedding_size)
        temp_emb.weight.data[:ple_max_len] = embeddings.position_embeddings.weight.data
        temp_emb.weight.data[ple_max_len:] = embeddings.position_embeddings.weight.data[-1][None,:].repeat(max_len-ple_max_len, 1)

        ple.embeddings.position_embeddings = temp_emb
        ple.config.max_position_embeddings = max_len

        ple.embeddings.position_ids = torch.arange(max_len).expand((1, -1))
        ple.embeddings.token_type_ids = torch.zeros(max_len, dtype=torch.long).expand((1, -1))        
    
    #Update config.emb_dim for process afterward
    config.emb_dim = ple.config.embedding_size

    for param in ple.parameters():
        param.requires_grad = False

    return ple


class Mapping(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_ratio=0.1):
        super(Mapping, self).__init__()            
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout_ratio)        

    def forward(self, x):
        return self.dropout(self.linear(x))



class PositionwiseFeedForward(nn.Module):
    def __init__(self, config):
        super(PositionwiseFeedForward, self).__init__()
        
        self.w_1 = nn.Linear(config.hidden_dim, config.pff_dim)
        self.w_2 = nn.Linear(config.pff_dim, config.hidden_dim)
        
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.dropout1 = nn.Dropout(config.dropout_ratio)
        self.dropout2 = nn.Dropout(config.dropout_ratio)


    def forward(self, x):
        x = self.norm(x)
        x = self.w_2(self.dropout1(F.gelu(self.w_1(x))))
        return self.dropout2(x)



class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout_ratio)
        self.mha = nn.MultiheadAttention(
            config.hidden_dim, 
            config.n_heads, 
            batch_first=True
        )

    def forward(self, q, k, v, key_padding_mask=None, attn_mask=None):
        #그럼 이때 norm의 대상이 되는 값이 뭐냐;;;
        x = self.norm(x)
        x = self.mha(query=q)[0]
        return self.dropout(x)
