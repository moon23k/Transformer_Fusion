from tqdm import tqdm
import torch, time, evaluate
from module.searc import Search



class Tester:
    def __init__(self, config, model, tokenizer, test_dataloader):
        super(Tester, self).__init__()
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = config.device
        self.dataloader = test_dataloader

        self.max_len = config.max_len
        self.beam_size = config.beam_size
        self.model_type = config.model_type

        if self.model_type != 'enc_dec':
            self.search = Search(config, model)


    @staticmethod
    def measure_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_min = int(elapsed_time / 60)
        elapsed_sec = int(elapsed_time - (elapsed_min * 60))
        return f"{elapsed_min}m {elapsed_sec}s"


    def test(self):
        self.model.eval()
        metric_module = evaluate.load('bleu')
        
        start_time = time.time()
        with torch.no_grad():
            for batch in tqdm(self.dataloader):   
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                                
                if self.model_type != 'enc_dec':
                    greedy_preds = self.search.greedy_search(pred, labels)
                    beam_preds = self.search.beam_search(pred, labels)

                else:
                    greedy_preds = self.model.generate(input_ids, attention_mask, max_new_tokens=self.max_len, use_cache=True)
                    beam_preds = self.model.generate(input_ids, attention_mask, num_beams=self.beam_size, max_new_tokens=self.max_len, use_cache=True)

                greedy_preds = self.tokenizer.decode(greedy_preds, skip_special_tokens=True)
                beam_preds = self.tokenizer.decode(beam_preds, skip_special_tokens=True)

                metric_module.add_batch(predictions=preds, 
                                        references=[[l] for l in labels])    

        bleu_score = metric_module.compute()['bleu'] * 100

        print('Test Results')
        print(f"  >> BLEU Score: {bleu_score:.2f}")
        print(f"  >> Spent Time: {self.measure_time(start_time, time.time())}")
        