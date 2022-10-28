import numpy as np
import os, random, argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from transformers import BertTokenizerFast

from models.transformer import Transformer
from models.downstream import FusedModel, SimpleModel

from modules.test import Tester
from modules.train import Trainer
from modules.inference import Translator
from modules.data import load_dataloader



def set_seed(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True



class Config(object):
    def __init__(self, args):
        #Args attrs
        self.task = args.task
        self.model_name = args.model
        self.scheduler = args.scheduler

        #Tokenizer attrs
        self.pad_idx = 0
        self.bos_idx = 101
        self.eos_idx = 102

        #Model attrs
        self.n_layers = 6
        self.n_heads = 12
        self.act = 'gelu'
        self.emb_dim = 768
        self.pff_dim = 3072
        self.hidden_dim = 768
        self.dropout_ratio = 0.1
        self.input_dim = 28996
        self.output_dim = 28996

        #Training attrs
        self.clip = 1
        self.n_epochs = 10
        self.batch_size = 32
        self.learning_rate = 5e-4
        self.ckpt_path = f"ckpt/{self.model_name}.pt"

        if self.task == 'inference':
            self.search = args.search
            self.device = torch.device('cpu')
        else:
            self.search = None
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")



def init_xavier(model):
    if hasattr(model, 'weight') and model.weight.dim() > 1:
        nn.init.xavier_uniform_(model.weight.data)



def count_params(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params
    

def check_size(model):
    param_size, buffer_size = 0, 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def load_tokenizer():
    return BertTokenizerFast.from_pretrained('distilbert-base-cased')
    


def load_model(config):
    #apply diff func according to the model
    if config.model_name == 'transformer':
        model = Transformer(config)
        model.apply(init_xavier)

    elif config.model_name == 'bert_simple':
        model = SimpleModel(config)
        model.apply(init_xavier)

    elif config.model_name == 'bert_fused':
        model = FusedModel(config)
        model.apply(init_xavier)
        
    if config.task != 'train':
        assert os.path.exists(config.ckpt_path)
        model_state = torch.load(config.ckpt_path, map_location=config.device)['model_state_dict']
        model.load_state_dict(model_state)

    print(f"The {config.model_name} model has loaded")
    print(f"--- Model Params: {count_params(model):,}")
    print(f"--- Model  Size : {check_size(model):.3f} MB\n")
    
    return model.to(config.device)



def main(config):
    model = load_model(config)

    if config.task == 'train': 
        trainer = Trainer(config, model)
        trainer.train()
    
    elif config.task == 'test':
        tester = Tester(config, model)
        tester.test()
    
    elif config.task == 'inference':
        translator = Translator(model, config)
        translator.translate()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', required=True)
    parser.add_argument('-model', required=True)
    parser.add_argument('-scheduler', default='constant', required=False)
    parser.add_argument('-search', default='greedy', required=False)
    
    args = parser.parse_args()
    assert args.task in ['train', 'test', 'inference']
    assert args.model in ['transformer', 'bert_simple', 'bert_fused']
 
    set_seed()
    config = Config(args)
    main(config)