import os, argparse, yaml, torch
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing
from transformers import set_seed, AutoTokenizer
from module import (
    load_model, load_dataloader,
    Trainer, Tester, SeqGenerator
)




class Config(object):
    def __init__(self, args, yaml_path='config.yaml'):
        
        self._set_attrs_from_args(args)
        self._set_params_from_yaml(yaml_path)

        self.max_len = self.max_len * 2 \
                if self.task == 'summarization' else self.max_len
        self.mname = f'{args.fusion_type}_{args.fusion_part}' \
                if args.fusion_type != 'simple' else f"{args.fusion_type}"
        
        self.ckpt = f"ckpt/{self.task}/{self.mname}_model.pt"
        self.tokenizer_path = f'data/{self.task}/tokenizer.json'
    
    def _set_params_from_yaml(self, yaml_path):
        with open(yaml_path, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        for group in params.values():
            for key, val in group.items():
                setattr(self, key, val)

    def _set_attrs_from_args(self, args):
        for key, val in vars(args).items():
            setattr(self, key, val)

    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")




def load_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(
        config.ple_name, model_max_length=config.max_len
    )

    #update config attrs
    setattr(config, 'vocab_size', tokenizer.vocab_size)
    setattr(config, 'pad_id', tokenizer.pad_token_id)
    setattr(config, 'bos_id', tokenizer.cls_token_id)
    setattr(config, 'eos_id', tokenizer.sep_token_id)        
    return tokenizer




def main(args):
    set_seed(42)
    config = Config(args)
    tokenizer = load_tokenizer(config)
    model = load_model(config)


    if config.mode == 'train':
        train_dataloader = load_dataloader(config, tokenizer, 'train')
        valid_dataloader = load_dataloader(config, tokenizer, 'valid')
        trainer = Trainer(config, model, train_dataloader, valid_dataloader)
        trainer.train()
    
    elif config.mode == 'test':
        test_dataloader = load_dataloader(config, tokenizer, 'test')
        tester = Tester(config, model, tokenizer, test_dataloader)
        tester.test()    
    
    elif config.mode == 'inference':
        generator = SeqGenerator(config, model, tokenizer)
        generator.inference()
    



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True)
    parser.add_argument('-fusion_type', required=True)
    parser.add_argument('-fusion_part', default='encoder', required=False)
    parser.add_argument('-search', default='greedy', required=False)    
    
    args = parser.parse_args()

    assert args.mode.lower() in ['train', 'test', 'inference']
    assert args.fusion_type.lower() in ['simple', 'parallel', 'sequential']
    assert args.fusion_part.lower() in ['encoder', 'decoder', 'encoder_decoder']
    assert args.search.lower() in ['greedy', 'beam']

    if args.mode != 'train':
        mname = f'{args.fusion_type}_{args.fusion_part}' \
                if args.fusion_type != 'simple' else f"{args.fusion_type}"
        assert os.path.exists(f'ckpt/{args.task}/{mname}_model.pt')    

    main(args)