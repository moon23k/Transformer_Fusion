import os, argparse, yaml, torch
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing
from transformers import set_seed, AutoTokenizer
from module import (
    load_model, load_dataloader,
    Trainer, Tester, SeqGenerator
)




class Config(object):
    def __init__(self, args):    

        with open('config.yaml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            for group in params.keys():
                for key, val in params[group].items():
                    setattr(self, key, val)

        self.task = args.task
        self.mode = args.mode
        self.mname = f'{args.fusion_type}_{args.fusion_part}'
        self.search_method = args.search
        self.ckpt = f"ckpt/{self.task}/{self.mname}_model.pt"
        self.tokenizer_path = f'data/{self.task}/tokenizer.json'

        use_cuda = torch.cuda.is_available()
        device_condition = use_cuda and self.mode != 'inference'
        self.device_type = 'cuda' if device_condition else 'cpu'
        self.device = torch.device(self.device_type)


    def update_attr(self, tokenizer):
        setattr(self, 'vocab_size', tokenizer.vocab_size)
        setattr(self, 'pad_id', tokenizer.pad_token_id)
        setattr(self, 'bos_id', tokenizer.cls_token_id)
        setattr(self, 'eos_id', tokenizer.sep_token_id)


    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")



def main(args):
    set_seed(42)
    config = Config(args)
    tokenizer = AutoTokenizer.from_pretrained(
        config.mname, model_max_length=config.max_len
    )
    config.update_attr(tokenizer)    
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
        generator = Generator(config, model, tokenizer)
        generator.inference()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', required=True)
    parser.add_argument('-mode', required=True)
    parser.add_argument('-fusion_type', required=True)
    parser.add_argument('-fusion_part', required=True)
    parser.add_argument('-search', default='greedy', required=False)    
    
    args = parser.parse_args()

    assert args.task in ['translation', 'dialogue', 'summarization']
    assert args.mode in ['train', 'test', 'inference']
    assert args.fusion_type in ['simple', 'parallel', 'sequential']
    assert args.fusion_part in ['encoder', 'decoder', 'encoder_decoder']
    assert args.search in ['greedy', 'beam']

    if args.mode == 'train':
        os.makedirs(f"ckpt/{args.task}", exist_ok=True)
    else:
        mname = f'{args.fusion_type}_{args.fusion_part}'
        assert os.path.exists(f'ckpt/{args.task}/{mname}_model.pt')    

    main(args)