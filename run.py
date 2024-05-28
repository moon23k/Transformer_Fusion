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

        self.mode = args.mode
        self.fusion_type = args.fusion_type
        self.fusion_part = args.fusion_part
        self.search = args.search

        self.enc_fuse = 'enc' in self.fusion_part
        self.dec_fuse = 'dec' in self.fusion_part
        self.mname = f'{args.fusion_type}_{args.fusion_part}' \
                if args.fusion_type != 'simple' else f"{args.fusion_type}"
        self.ckpt = f"ckpt/{self.mname}_model.pt"


        use_cuda = torch.cuda.is_available()
        device_condition = use_cuda and self.mode != 'inference'
        self.device_type = 'cuda' if device_condition else 'cpu'
        self.device = torch.device(self.device_type)    


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
        train_dataloader = load_dataloader(tokenizer, 'train', config.batch_size)
        valid_dataloader = load_dataloader(tokenizer, 'valid', config.batch_size)
        trainer = Trainer(config, model, train_dataloader, valid_dataloader)
        trainer.train()
    
    elif config.mode == 'test':
        test_dataloader = load_dataloader(tokenizer, 'test', config.batch_size)
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
        assert os.path.exists(f'ckpt/{mname}_model.pt')    

    main(args)