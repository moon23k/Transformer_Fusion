import os, argparse, torch

from module.test import Tester
from module.train import Trainer
from module.data import load_dataloader

from model.fused import FusedModel
from model.simple import SimpleModel

from transformers import (set_seed,
                          BertTokenizerFast,
                          BertGenerationDecoder,
                          BertGenerationEncoder,
                          EncoderDecoderModel)



class Config(object):
    def __init__(self, args):    

        self.task = args.task
        self.mode = args.mode
        self.model_type = args.model
        self.bert_mname = 'prajjwal1/bert-small'
        self.ckpt = f"ckpt/{self.task}_{self.model_name}.pt"
        self.src, self.trg = self.task[:2], self.task[2:]
        
        self.clip = 1
        self.max_len = 300
        self.n_epochs = 10
        self.batch_size = 16
        self.learning_rate = 5e-4
        self.iters_to_accumulate = 4

        self.n_heads = 8
        self.n_layers = 3
        self.pff_dim = 512
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



def load_model(config):
    if config.model_type == 'simple':
        model = SimpleModel(config)

    elif config.model_type == 'fused':
        model = FusedModel(config)
    
    elif config.model_type == 'generation':
        encoder = BertGenerationEncoder.from_pretrained(config.bert_mname, 
                                                        bos_token_id=config.bos_id,
                                                        eos_token_id=config.eos_id)
        decoder = BertGenerationDecoder.from_pretrained(config.bert_mname, 
                                                        add_cross_attention=True, 
                                                        is_decoder=True,
                                                        bos_token_id=config.bos_id,
                                                        eos_token_id=config.eos_id)
        model = EncoderDecoderModel(encoder=encoder, decoder=decoder)        
        model.config.decoder.decoder_start_token_id = config.bos_id
        model.config.pad_token_id = config.pad_id
        model.config.vocab_size = config.vocab_size

    print(f"BERT {config.model_name.upper()} Model for has loaded")

    
    if config.mode != 'train':
        assert os.path.exists(config.ckpt)
        model_state = torch.load(config.ckpt, map_location=config.device)['model_state_dict']
        model.load_state_dict(model_state)
        print(f"Trained Model States has loaded on the Model")

    
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

    
    print(f"--- Model Params: {count_params(model):,}")
    print(f"--- Model  Size : {check_size(model):.3f} MB\n")
    
    return model.to(config.device)



def train(config, model):
    train_dataloader = load_dataloader(config, 'train')
    valid_dataloader = load_dataloader(config, 'valid')
    trainer = Trainer(config, model, train_dataloader, valid_dataloader)
    trainer.train()


def test(config, model, tokenizer):
    test_dataloader = load_dataloader(config, 'test')
    tester = Tester(config, model, tokenizer, test_dataloader)
    tester.test()    



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
    tokenizer = BertTokenizerFast.from_pretrained(config.bert_mname, model_max_length=300)

    setattr(config, 'vocab_size', tokenizer.vocab_size)
    setattr(config, 'pad_id', tokenizer.pad_token_id)
    setattr(config, 'bos_id', tokenizer.cls_token_id)
    setattr(config, 'eos_id', tokenizer.sep_token_id)

    model = load_model(config)    

    if config.mode == 'train':
        train(config, model)
    
    elif config.mode == 'test':
        test(config, model, tokenizer)
    
    elif config.mode == 'inference':
        inference(model, tokenizer)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', required=True)
    parser.add_argument('-mode', required=True)
    parser.add_argument('-model', required=True)    
    
    args = parser.parse_args()

    assert args.task in ['ende', 'deen']
    assert args.mode in ['train', 'test', 'inference']
    assert args.model in ['simple', 'fused', 'generation']    

    main(args)