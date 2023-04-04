from tqdm import tqdm
import torch, time, evaluate



class Tester:
    def __init__(self, config, model, tokenizer, test_dataloader):
        super(Tester, self).__init__()
        
        self.model = model
        self.max_len = config.max_len
        self.tokenizer = tokenizer
        self.device = config.device
        self.dataloader = test_dataloader


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
                                
                preds = self.model.generate(input_ids, attention_mask, max_new_tokens=self.max_len, use_cache=True)

                metric_module.add_batch(predictions=preds, 
                                        references=[[l] for l in labels])    

        bleu_score = metric_module.compute()['bleu'] * 100

        print('Test Results')
        print(f"  >> BLEU Score: {bleu_score:.2f}")
        print(f"  >> Spent Time: {self.measure_time(start_time, time.time())}")