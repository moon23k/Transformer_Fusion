import torch



class Translator:
	def __init__(self, config, model, src_tokenizer, trg_tokenizer):
		self.model = model
		self.max_len = 100
		self.device = config.device
		self.search = config.search
		self.bos_idx = config.bos_idx
		self.src_tokenizer = src_tokenizer
		self.trg_tokenizer = trg_tokenizer
		self.model_name == config.model_name

	
	def tranaslate(self, config):
		self.model.eval()
		print('Type "quit" to terminate Translation')
		while True:
			user_input = input('please type text >> ')
			if user_input == 'quit':
				print('--- Terminate the Translation ---')
				print('-' * 30)
				break

			src = self.src_tokenizer.Encode(user_input)
			src = torch.LongTensor(src).to(self.device)
			pred_seq = torch.LongTensor([self.bos_idx]).to(self.device)

			for t in range(self.max_len):
				out = self.model(src, pred_seq)
				pred_word = out.argmax(-1)
				pred_seq = torch.cat([pred_seq, pred_word])

			print(f"Original Sentence:   {user_input}")
			print(f'Translated Sequenec: {self.trg_tokenizer.Decode(pred_seq)}\n')
			
			
import torch
import argparse
import sentencepiece as spm
from sacremoses import MosesTokenizer

from utils.util import Config, set_seed
from models.transformer.module import create_src_mask, create_trg_mask




def transformer_run(model, tokenizer, config, max_tokens=100):
	with torch.no_grad():
		print('Type "quit" to terminate Translation')
	    while True:
	        seq = input('\nUser Input sentence >> ')
	        if seq == 'quit':
	            print(' --- Terminate the Translation! ---')
	            break
	        
	        #Tokenize user Input with Moses
	        mt = MosesTokenizer(lang='en')
	        src = mt.tokenize(src)

	        #Convert tokens to ids with sentencepiece vocab
	        src = tokenizer.EncodeAsIds(seq)

	        #Convert ids to tensor
	        src = torch.tensor(src, dtype=torch.long).unsqueeze(0).to(config.device)
	        
	        src_mask = create_src_mask(src)
	        src = model.embedding(src)
	        enc_out = model.encoder(src, src_mask)
	        trg_indice = [tokenizer.bos_id()]

            for _ in range(max_tokens):
                trg_tensor = torch.tensor(trg_indice, dtype=torch.long).unsqueeze(0).to(config.device)
                trg_mask = create_trg_mask(trg_tensor)

                trg = model.embedding(trg_tensor)

                dec_out, _ = model.decoder(enc_out, trg, src_mask, trg_mask)
                out = model.fc_out(dec_out)

                pred_token = out.argmax(2)[:, -1].item()
                trg_indice.append(pred_token)

                if pred_token == tokenizer.eos_id():
                    break
	            
	        pred_seq = trg_indice[1:]
	        pred_seq = tokenizer.Decode(pred_seq)

	        print(f"Translated sentence >> {pred_seq}")



def seq2seq_run(model, tokenizer, config, max_tokens=100):
	with torch.no_grad():
		print('Type "quit" to terminate Translation')
	    while True:
	        seq = input('\nUser Input sentence >> ')
	        if seq == 'quit':
	            print(' --- Terminate the Translation! ---')
	            print('------------------------------------')
	            break
	        
	        #Tokenize user Input with Moses
	        mt = MosesTokenizer(lang='en')
	        src = mt.tokenize(src)

	        #Convert tokens to ids with sentencepiece vocab
	        src = tokenizer.EncodeAsIds(seq)
	        if config.model == 'seq2seq':
	        	src[1:-1] = seq[-2:0:-1]

	        #Convert ids to tensor
	        src = torch.tensor(src, dtype=torch.long).unsqueeze(0)
	        
	        src = model.embedding(src)
	        enc_out = model.encoder(src, src_mask)
	        trg_indice = [tokenizer.bos_id()]

	        for _ in range(max_tokens):
	            trg_tensor = torch.tensor(trg_indice, dtype=torch.long).unsqueeze(0)
	            trg = model.embedding(trg)

                dec_out, _ = model.decoder()
                out = model.fc_out(dec_out)

	            pred_token = out.argmax(2)[:, -1].item()
	            trg_indice.append(pred_token)

	            if pred_token == tokenizer.eos_id():
	                break
	        
	        pred_seq = trg_indice[1:]
	        pred_seq = tokenizer.Decode(pred_seq)

	        print(f"Translated sentence >> {pred_seq}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', required=True)
    args = parser.parse_args()
    
    assert args.bert in ['seq2seq', 'attention', 'transformer']
    
    set_seed()
    config = Config(args)
    config.device = torch.device('cpu')

    #Load Model
	model = load_model(config)
	model_state = torch.load()
	model.load_state_dict(model_state)
	model.eval()

	#Load Tokenizer
	tokenizer = spm.SentencePieceProcessor()
	tokenizer.load('data/vocab/spm.model')
	tokenizer.SetEncodeExtraOptions('bos:eos')


	if config.model == 'transformer':
		transformer_run(model, tokenizer, config)
	else:
		seq2seq_run(model, tokenizer, config)
