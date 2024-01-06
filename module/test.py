import torch, evaluate



class Tester:
    def __init__(self, config, model, tokenizer, test_dataloader):
        super(Tester, self).__init__()
        
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = test_dataloader

        self.task = config.task
        self.bos_id = config.bos_id
        self.eos_id = config.eos_id
        self.device = config.device
        self.max_len = config.max_len
        self.model_type = config.model_type
        
        self.metric_name = 'BLEU' if self.task == 'translation' else 'ROUGE'
        self.metric_module = evaluate.load(self.metric_name.lower())



    def test(self):
        score = 0.0         
        self.model.eval()

        with torch.no_grad():
            for batch in self.dataloader:
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                pred = self.model.generate(input_ids, attention_mask)
                
                pred = self.tokenize(pred)
                labels = self.tokenize(labels)

                score += self.evaluate(pred, labels)

        txt = f"TEST Result on {self.task.upper()} with {self.model_type.upper()} model"
        txt += f"\n-- Score: {round(score/len(self.dataloader), 2)}\n"
        print(txt)


    def tokenize(self, batch):
        return [self.tokenizer.decode(x) for x in batch.tolist()]





    def evaluate(self, pred, label):
        if all(elem == '' for elem in pred):
            return 0.0

        #For Transaltion Evaluation
        if self.task == 'translation':
            score = self.metric_module.compute(
                predictions=pred, 
                references =[[l] for l in label]
            )['bleu']
        #For Dialgue & Summarization Evaluation
        else:
            score = self.metric_module.compute(
                predictions=pred, 
                references =[[l] for l in label]
            )['rouge2']

        return score * 100        