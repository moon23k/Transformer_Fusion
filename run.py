import os, argparse, torch
from transformers import set_seed, AutoTokenizer
from module import (
    load_dataloader,
    load_model,
    Trainer, 
    Tester,
    Generator
)




class Config(object):
    def __init__(self, args):    

        self.mode = args.mode
        self.model_type = args.model
        self.plm_mname = 'albert-base-v2'
        self.ckpt = f"ckpt/{self.model_type}_model.pt"

        use_cuda = torch.cuda.is_available()
        self.device_type = 'cuda' if use_cuda else 'cpu'

        if self.mode == 'inference':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if use_cuda else 'cpu')

        os.makedirs(f'ckpt/{self.task}', exist_ok=True)


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
    config = Config(args.task, args.task)
    tokenizer = AutoTokenizer.from_pretrained(
        config.plm_mname, model_max_length=config.max_len
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
    parser.add_argument('-model', required=True)
    parser.add_argument('-search', default='greedy', required=False)    
    
    args = parser.parse_args()

    assert args.task in ['translation', 'dialogue', 'summarization']
    assert args.mode in ['train', 'test', 'inference']
    assert args.model in ['simple', 'fused', 'enc_dec']
    assert args.model in ['greedy', 'beam']

    

    main(args)