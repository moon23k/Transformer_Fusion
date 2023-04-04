import os, argparse, torch

from module.test import Tester
from module.train import Trainer
from module.model import load_model
from module.data import load_dataloader

from transformers import (set_seed,
                          BertTokenizerFast,
                          BertGenerationDecoder,
                          BertGenerationEncoder,
                          EncoderDecoderModel)



class Config(object):
    def __init__(self, args):    

        self.mode = args.mode
        self.model_type = args.model
        self.bert_mname = 'prajjwal1/bert-small'
        self.ckpt = f"ckpt/{self.task}_{self.model_type}.pt"
        
        self.clip = 1
        self.lr = 5e-5
        self.max_len = 300
        self.n_epochs = 10
        self.batch_size = 128
        self.iters_to_accumulate = 4

        self.n_heads = 8
        self.n_layers = 3
        self.pff_dim = 2048
        self.hidden_dim = 512
        self.dropout_ratio = 0.1
        self.emb_dim = self.hidden_dim // 2

        use_cuda = torch.cuda.is_available()
        self.device_type = 'cuda' if use_cuda else 'cpu'

        if self.mode == 'inference':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if use_cuda else 'cpu')


    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")




def inference(model, tokenizer):
    model.eval()
    print(f'--- Inference Process Started! ---')
    print('[ Type "quit" on user input to stop the Process ]')
    
    while True:
        input_seq = input('\nUser Input Sequence >> ').lower()

        #End Condition
        if input_seq == 'quit':
            print('\n--- Inference Process has terminated! ---')
            break        

        #convert user input_seq into model input_ids
        encodings = tokenizer(input_seq)

        if isinstance(model, EncoderDecoderModel):
            preds = model.generate(**encodings, use_cache=True)
        else:
            preds = model.generate(**encodings)

        preds = tokenizer.decode(preds, skip_special_tokens=True)

        #Search Output Sequence
        print(f"Model Out Sequence >> {preds}")       




def main(args):
    set_seed(42)
    config = Config(args.task, args.task)
    tokenizer = BertTokenizerFast.from_pretrained(config.bert_mname, 
                                                  model_max_length=config.max_len)

    setattr(config, 'vocab_size', tokenizer.vocab_size)
    setattr(config, 'pad_id', tokenizer.pad_token_id)
    setattr(config, 'bos_id', tokenizer.cls_token_id)
    setattr(config, 'eos_id', tokenizer.sep_token_id)

    model = load_model(config)    

    if config.mode == 'train':
        train_dataloader = load_dataloader(config, 'train')
        valid_dataloader = load_dataloader(config, 'valid')
        trainer = Trainer(config, model, train_dataloader, valid_dataloader)
        trainer.train()
    
    elif config.mode == 'test':
        test_dataloader = load_dataloader(config, 'test')
        tester = Tester(config, model, tokenizer, test_dataloader)
        tester.test()    
    
    elif config.mode == 'inference':
        inference(model, tokenizer)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True)
    parser.add_argument('-model', required=True)    
    
    args = parser.parse_args()

    assert args.mode in ['train', 'test', 'inference']
    assert args.model in ['simple', 'fused', 'enc_dec']    

    main(args)