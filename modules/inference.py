import torch
from run import load_tokenizer
from modules.search import Search



class Translator:
	def __init__(self, config, model):
		self.model = model
        if self.model.training:
            self.model.eval()

		self.device = config.device
		self.search = config.search
		self.bos_idx = config.bos_idx
		self.model_name == config.model_name
        self.tokenizer = load_tokenizer(config)
        self.search = Search(config, self.model)            

	
	def tranaslate(self, config):
		print('Type "quit" to terminate Translation')
		while True:
			user_input = input('Please Type Text >> ')
			if user_input.lower() == 'quit':
				print('--- Terminate the Translation ---')
				print('-' * 30)
				break

			src = self.tokenizer.Encode(user_input)
			src = torch.LongTensor(src).unsqueeze(0).to(self.device)

            if self.search == 'beam':
                pred_seq = self.search.beam_search(src)
            elif self.search == 'greedy':
                pred_seq = self.search.greedy_search(src)

			print(f"Original   Sequence: {user_input}")
			print(f'Translated Sequence: {self.tokenizer.Decode(pred_seq)}\n')
			
