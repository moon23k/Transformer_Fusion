import numpy as np
import os, yaml, random, argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from modules.data import load_dataloader
from models.downstream import FusedModel, AddTopModel

from modules.test import Tester
from modules.train import Trainer
from modules.inference import Translator

import transformers as T


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
        with open('configs/model.yaml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            params = params[args.model]
            for p in params.items():
                setattr(self, p[0], p[1])

        self.task = args.task
        self.model_name = args.model
        self.scheduler = args.scheduler
        
        self.unk_idx = 0
        self.pad_idx = 1
        self.bos_idx = 2
        self.eos_idx = 3

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


def load_tokenizer(model_name):
    if model_name == 'bert':
        return T.BertTokenizer.from_pretrained()
    elif model_name == 'albert':
        return T.BertTokenizer.from_pretrained()
    elif model_name == 'distil_bert':
        return T.BertTokenizer.from_pretrained()
    

    return tokenizer


def load_model(config):

    if config.model_name == 'transformer':
        model = Transformer(config)
        model.apply(init_xavier)
        
    if config.task != 'train':
        assert os.path.exists(config.ckpt_path)
        model_state = torch.load(config.ckpt_path, map_location=config.device)['model_state_dict']
        model.load_state_dict(model_state)

    print(f"The {config.model_name} model has loaded")
    print(f"--- Model Params: {count_params(model):,}")
    print(f"--- Model  Size : {check_size(model):.3f} MB")
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
    assert args.model in ['bert', 'gpt', 'bart']
 
    set_seed()
    config = Config(args)
    main(config)