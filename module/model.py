import os, torch
import torch.nn as nn
from transformers import AutoModel
from model import SimpleModel, FusionModel



    
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



def init_weights(model):
    for name, param in model.named_parameters():
        if 'ple' not in name and 'weight' in name and 'norm' not in name:
            nn.init.xavier_uniform_(param)



def load_model(config):
    if config.model_type == 'simple':
        model = SimpleModel(config)
    elif config.model_type == 'fusion':
        model = FusionModel(config)

    init_weights(model)
    print(f"Initialized {config.model_type.upper()} model has loaded")
    

    if config.mode != 'train':
        assert os.path.exists(config.ckpt)
        model_state = torch.load(config.ckpt, map_location=config.device)['model_state_dict']
        model.load_state_dict(model_state)
        print(f"Trained Model States has loaded on the Model")
    '''
    ple_keys = [key for key in pretrained_model_state.keys() if 'ple' not in key]
    for key in ple_keys:
        current_model_state[key] = pretrained_model_state[key]
    '''


    print_model_desc(model)
    return model.to(config.device)