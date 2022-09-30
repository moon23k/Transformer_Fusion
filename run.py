import numpy as np
import yaml, random, argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from models.base import BaseModel
from models.simple import SimpleModel
from models.fused import FusedModel

from modules.test import Tester
from modules.train import Trainer
from modules.inference import Translator
from modules.data import load_dataloader

from transformers import BertTokenizer




class Config(object):
    def __init__(self, args):    
        with open('models/configs.yaml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)            
        for p in params.items():
            setattr(self, p[0], p[1])

        self.task = args.task
        self.model_name = args.model
        
        self.unk_idx = 0
        self.pad_idx = 1
        self.bos_idx = 2
        self.eos_idx = 3

        self.clip = 1
        self.n_epochs = 1
        self.batch_size = 128
        self.learning_rate = 1e-4

        if self.task == 'inference':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")



def set_seed(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True



def load_model(config):
    def init_xavier(model):
        if hasattr(model, 'weight') and model.weight.dim() > 1:
            nn.init.xavier_uniform_(model.weight.data)

    if config.model_name == 'base':
        model = BaseModel(config)
        model.apply(init_xavier)
    elif config.model_name == 'simple':
        model = SimpleModel(config)
        model.apply(init_xavier)
    elif config.model_name == 'fused':
        model = FusedModel(config)
        model.apply(init_xavier)
        
    if config.task != 'train':
        model_state = torch.load(config.ckpt_path, map_location=config.device)['model_state_dict']
        model.load_state_dict(model_state)

    return model.to(config.device)


def load_tokenizer(side):
    assert side in ['src', 'trg']

    if side == 'src':
        return BertTokenizer.from_pretrained('bert-base-cased')
    else:
        return BertTokenizer.from_pretrained('bert-base-german-cased')


def main(config):
    model = load_model(config)

    if config.task == 'train':
        train_dataloader = load_dataloader(config, 'train')
        valid_dataloader = load_dataloader(config, 'valid')        
        trainer = Trainer(config, model, train_dataloader, valid_dataloader)
        trainer.train()
    
    elif config.task == 'test':
        test_dataloader = load_dataloader(config, 'test')
        trg_tokenizer = load_tokenizer('trg')
        tester = Tester(config, model, test_dataloader, trg_tokenizer)
        tester.test()
    
    elif config.task == 'inference':
        src_tokenizer = load_tokenizer('src')
        trg_tokenizer = load_tokenizer('trg')
        translator = Translator(config, model, src_tokenizer, trg_tokenizer)
        translator.translate()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', required=True)
    parser.add_argument('-model', required=True)
    parser.add_argument('-scheduler', required=False)
    
    args = parser.parse_args()
    assert args.task in ['train', 'test', 'inference']
    assert args.model in ['base', 'simple', 'fused']
 
    set_seed()
    config = Config(args)
    main(config)