import os, torch
import torch.nn as nn
from transformers import AutoModel
from model import SimpleModel, ParallelModel, SequentialModel




def init_weights(model):
    for name, param in model.named_parameters():
        if 'ple' not in name and 'weight' in name and 'norm' not in name:
            nn.init.xavier_uniform_(param)


    
def print_model_desc(model):
    #Number of trainerable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"--- Model Total Params: {total_params:,}")
    print(f"--- Model Trainable Params: {trainable_params:,}")

    #Model size check
    param_size, buffer_size = 0, 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"--- Model  Size : {size_all_mb:.3f} MB\n")



def load_ple(config):
    ple = AutoModel.from_pretrained(config.ple_name)

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



def load_model(config):
    ple = load_ple(config)

    if 'simple' in config.mname:
        model = SimpleModel(config, ple)
    elif 'parallel' in config.mname:
        model = ParallelModel(config, ple)
    elif 'sequential' in config.mname:
        model = SequentialModel(config, ple)

    init_weights(model)
    print(f"Initialized {config.mname.upper()} model has loaded")
    
    if config.mode != 'train':
        assert os.path.exists(config.ckpt)
        model_state = torch.load(config.ckpt, map_location=config.device)['model_state_dict']
        model.load_state_dict(model_state)
        print(f"Trained Model States has loaded on the Model")

    print_model_desc(model)
    return model.to(config.device)