import torch, evaluate




class Tester:
    def __init__(self, config, model, tokenizer, test_dataloader):
        super(Tester, self).__init__()
        
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = test_dataloader
        self.metric_module = evaluate.load('bleu')

        self.mname = config.mname
        self.pad_id = config.pad_id
        self.bos_id = config.bos_id
        self.eos_id = config.eos_id
        self.device = config.device
        self.max_len = config.max_len

        self.enc_fuse = config.enc_fuse
        self.dec_fuse = config.dec_fuse
        self.fusion_type = config.fusion_type
        


    def test(self):
        score = 0.0
        self.model.eval()

        with torch.no_grad():
            for batch in self.dataloader:
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                if self.fusion_type == 'simple':
                    pred = self.simple_predict(input_ids, attention_mask)
                else:
                    pred = self.fusion_predict(input_ids, attention_mask)            

                score += self.evaluate(pred, labels)

        txt = f"TEST Result on {self.mname.upper()} model"
        txt += f"\n-- Score: {round(score/len(self.dataloader), 2)}\n"
        print(txt)                




    def simple_predict(self, input_ids, attention_mask):
        #Prerequisites
        batch_size = input_ids.size(0)
        pred = torch.zeros((batch_size, self.max_len), dtype=torch.long)
        pred = pred.fill_(self.pad_id).to(self.device)
        pred[:, 0] = self.bos_id

        e_mask = self.model.pad_mask(input_ids)
        memory = self.model.encoder(input_ids, attention_mask)

        #Decoding
        for idx in range(1, self.max_len):
            y = pred[:, :idx]
            y = self.model.encoder.ple.embeddings(y)
            d_out = self.model.decoder(y, memory, e_mask, None)
            logit = self.model.generator(d_out)
            pred[:, idx] = logit.argmax(dim=-1)[:, -1]

            #Early Stop Condition
            if (pred == self.eos_id).sum().item() == batch_size:
                break

        return pred
    



    def fusion_predict(self, input_ids, attention_mask):
        #Prerequisites
        batch_size = input_ids.size(0)
        pred = torch.zeros((batch_size, self.max_len), dtype=torch.long)
        pred = pred.fill_(self.pad_id).to(self.device)
        pred[:, 0] = self.bos_id

        x = input_ids
        x = self.model.ple.embeddings(x)
        e_mask = self.model.pad_mask(input_ids)
        p_proj = self.model.ple_project(input_ids, attention_mask)
        memory = self.model.encoder(x, p_proj if self.enc_fuse else None, e_mask)

        #Decoding
        for idx in range(1, self.max_len):
            y = pred[:, :idx]
            y = self.model.ple.embeddings(y)
            d_out = self.model.decoder(y, memory, p_proj if self.dec_fuse else None, e_mask, None)
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

        score = self.metric_module.compute(
            predictions=pred, 
            references =[[l] for l in label]
        )['bleu']

        return score * 100