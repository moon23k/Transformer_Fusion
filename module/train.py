import time, math, json, torch
import torch.nn as nn
import torch.amp as amp
import torch.optim as optim



class Trainer:
    def __init__(self, config, model, train_dataloader, valid_dataloader):
        super(Trainer, self).__init__()
        
        self.model = model
        self.src = config.src
        self.trg = config.trg
        self.task = config.task
        self.clip = config.clip
        self.device = config.device
        self.n_epochs = config.n_epochs

        self.device_type = config.device_type
        self.scaler = torch.cuda.amp.GradScaler()
        self.iters_to_accumulate = config.iters_to_accumulate        

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.optimizer, self.bert_optimizer = self.get_optims(config)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        
        self.ckpt = config.ckpt
        self.record_path = f"ckpt/{config.task}.json"
        self.record_keys = ['epoch', 'train_loss', 'train_ppl',
                            'valid_loss', 'valid_ppl', 
                            'learning_rate', 'train_time']



    def get_optims(self, config):
        if config.model_name == 'generation':
            optimizer = optim.Adam(params=self.model.parameters(), 
                                   lr=config.learning_rate * 0.1, 
                                   betas=(0.9, 0.98), 
                                   eps=1e-8)            
            bert_optimizer = None

        elif config.model_name != 'generation':
            optim_params = [self.model.encoder.parameters(),
                            self.model.decoder.parameters(),
                            self.model.fc_out.parameters()]
            
            optimizer = optim.Adam(params=optim_params, 
                                   lr=config.learning_rate, 
                                   betas=(0.9, 0.98), 
                                   eps=1e-8)

            bert_optimizer = optim.Adam(self.model.bert.parameters(), 
                                        lr=config.learning_rate * 0.1, 
                                        betas=(0.9, 0.98), 
                                        eps=1e-8)

        return optimizer, bert_optimizer
        


    def print_epoch(self, record_dict):
        print(f"""Epoch {record_dict['epoch']}/{self.n_epochs} | \
              Time: {record_dict['train_time']}""".replace(' ' * 14, ''))
        
        print(f"""  >> Train Loss: {record_dict['train_loss']:.3f} | \
              Train PPL: {record_dict['train_ppl']:.2f}""".replace(' ' * 14, ''))

        print(f"""  >> Valid Loss: {record_dict['valid_loss']:.3f} | \
              Valid PPL: {record_dict['valid_ppl']:.2f}\n""".replace(' ' * 14, ''))



    @staticmethod
    def measure_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_min = int(elapsed_time / 60)
        elapsed_sec = int(elapsed_time - (elapsed_min * 60))
        return f"{elapsed_min}m {elapsed_sec}s"



    def split_batch(self, batch):
        input_ids = batch[f'{self.src}_ids'].to(self.device)
        attention_mask =  batch[f'{self.src}_mask'].to(self.device)
        labels = batch[f'{self.trg}_ids'].to(self.device)
        
        return input_ids, attention_mask, labels



    def train(self):
        best_loss, records = float('inf'), []
        for epoch in range(1, self.n_epochs + 1):
            start_time = time.time()

            record_vals = [epoch, *self.train_epoch(), *self.valid_epoch(), 
                           self.optimizer.param_groups[0]['lr'],
                           self.measure_time(start_time, time.time())]
            record_dict = {k: v for k, v in zip(self.record_keys, record_vals)}
            
            records.append(record_dict)
            self.print_epoch(record_dict)
            
            val_loss = record_dict['valid_loss']
            self.scheduler.step(val_loss)

            #save best model
            if best_loss > val_loss:
                best_loss = val_loss
                torch.save({'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()},
                            self.ckpt)
            
        #save train_records
        with open(self.record_path, 'w') as fp:
            json.dump(records, fp)




    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        tot_len = len(self.train_dataloader)

        for idx, batch in enumerate(self.train_dataloader):
            input_ids, attention_mask, labels = self.split_batch(batch)

            with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                loss = self.model(input_ids = input_ids, 
                                  attention_mask = attention_mask,
                                  labels = labels)[0]
                loss = loss / self.iters_to_accumulate
            
            #Backward Loss
            self.scaler.scale(loss).backward()        
            
            if (idx + 1) % self.iters_to_accumulate == 0:
                #Gradient Clipping
                self.scaler.unscale_(self.optimizer)
                if self.model_name != 'generation':
                    self.scaler.unscale_(self.optimizer)

                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
                

                #Gradient Update & Scaler Update
                self.scaler.step(self.optimizer)
                if self.model_name != 'generation':
                    self.scaler.step(self.bert_optimizer)
                
                self.scaler.update()
                
                self.optimizer.zero_grad()
                if self.model_name != 'generation':
                    self.bert_optimizer.zero_grad()

            epoch_loss += loss.item()
        
        epoch_loss = round(epoch_loss / tot_len, 3)
        epoch_ppl = round(math.exp(epoch_loss), 3)    
        return epoch_loss, epoch_ppl
    


    def valid_epoch(self):
        self.model.eval()
        epoch_loss = 0
        tot_len = len(self.valid_dataloader)
        
        with torch.no_grad():
            for _, batch in enumerate(self.valid_dataloader):   
                input_ids, attention_mask, labels = self.split_batch(batch)           
                
                with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                    loss = self.model(input_ids = input_ids, 
                                      attention_mask = attention_mask,
                                      labels = labels)[0]

                epoch_loss += loss.item()
        
        epoch_loss = round(epoch_loss / tot_len, 3)
        epoch_ppl = round(math.exp(epoch_loss), 3)        
        return epoch_loss, epoch_ppl