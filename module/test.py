import torch, evaluate



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
        self.metric_module = evaluate.load('bleu')


    def test(self):
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(self.dataloader):   
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                                
                if self.model_type != 'enc_dec':
                    pred = self.model.predict(pred, labels)

                else:
                    pred = self.model.generate(
                        input_ids, attention_mask, 
                        max_new_tokens=self.max_len, use_cache=True
                    )
                
                pred = self.tokenizer.decode(
                    pred, skip_special_tokens=True
                )

                self.metric_module.add_batch(
                    predictions=preds, 
                    references=[[l] for l in labels]
                )    

        bleu_score = metric_module.compute()['bleu'] * 100

        print('Test Results')
        print(f"  >> BLEU Score: {bleu_score:.2f}")
        print(f"  >> Spent Time: {self.measure_time(start_time, time.time())}")
        