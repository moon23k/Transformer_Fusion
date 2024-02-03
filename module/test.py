import torch, evaluate




class Tester:
    def __init__(self, config, model, tokenizer, test_dataloader):
        super(Tester, self).__init__()
        
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = test_dataloader

        self.task = config.task
        self.pad_id = config.pad_id
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

                if self.model_type == 'simple':
                    pred = self.simple_predict(input_ids, attention_mask)
                elif self.model_type == 'fusion':
                    pred = self.fusion_predict(input_ids, attention_mask)            

                score += self.evaluate(pred, labels)

        txt = f"TEST Result on {self.task.upper()} with {self.model_type.upper()} model"
        txt += f"\n-- Score: {round(score/len(self.dataloader), 2)}\n"
        print(txt)                




    def simple_predict(self, input_ids, attention_mask):
        
        batch_size = input_ids.size(0)
        pred = torch.zeros((batch_size, self.max_len), dtype=torch.long)
        pred = pred.fill_(self.pad_id).to(self.device)
        pred[:, 0] = self.bos_id

        e_mask = self.model.mask(input_ids)
        memory = self.model.encode(input_ids, attention_mask)

        for idx in range(1, self.max_len):
            y = pred[:, :idx]
            d_out = self.model.decoder(y, memory, e_mask, None)
            logit = self.model.generator(d_out)
            pred[:, idx] = logit.argmax(dim=-1)[:, -1]

            #Early Stop Condition
            if (pred == self.eos_id).sum().item() == batch_size:
                break

        return pred
    



    def fusion_predict(self, input_ids, attention_mask):

        batch_size = input_ids.size(0)
        pred = torch.zeros((batch_size, self.max_len), dtype=torch.long)
        pred = pred.fill_(self.pad_id).to(self.device)
        pred[:, 0] = self.bos_id

        e_mask = self.model.mask(input_ids)

        ple_out = self.model.ple(
            input_ids=input_ids, 
            attention_mask=attention_mask
        ).last_hidden_state
        ple_out = self.model.mapping(ple_out)

        memory = self.model.encode(input_ids, ple_out, e_mask)

        for idx in range(1, self.max_len):
            y = pred[:, :idx]
            d_out = self.model.decoder(y, memory, ple_out, e_mask, None)
            logit = self.model.generator(d_out)
            pred[:, idx] = logit.argmax(dim=-1)[:, -1]

            #Early Stop Condition
            if (pred == self.eos_id).sum().item() == batch_size:
                break

        return pred




    def evaluate(self, pred, label):
        #Tokenize
        pred = [self.tokenizer.decode(x) for x in pred.tolist()]
        label = [self.tokenizer.decode(x) for x in label.tolist()]

        #End Condition
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