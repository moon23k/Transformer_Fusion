import torch.nn as nn
import math, time, torch
from modules.search import Search
from torchtext.data.metrics import bleu_score



class Tester(Search):
    def __init__(self, config, model, dataloader, tokenizer):
        super(Tester, self).__init__(config, model)
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = dataloader

        self.device = config.device
        self.pad_idx = config.pad_idx
        self.bos_idx = config.bos_idx
        self.eos_idx = config.eos_idx        
        self.model_name = config.model_name
        self.output_dim = config.output_dim
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx, label_smoothing=0.1).to(config.device)


    def get_bleu_score(self, pred, trg):
        score = 0
        batch_size = trg.size(0)
        
        for can, ref in zip(pred, trg.tolist()):
            score += bleu_score([self.tokenizer.Decode(can).split()],
                                [[self.tokenizer.Decode(ref).split()]])
        return (score / batch_size) * 100

    @staticmethod
    def measure_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_min = int(elapsed_time / 60)
        elapsed_sec = int(elapsed_time - (elapsed_min * 60))
        return f"{elapsed_min}m {elapsed_sec}s"


    def test(self):
        self.model.eval()
        tot_len = len(self.dataloader)
        tot_loss, tot_greedy_bleu, tot_beam_bleu = 0, 0, 0
        start_time = time.time()
        
        with torch.no_grad():
            for idx, batch in enumerate(self.dataloader):
                src, trg = batch['src'].to(self.device), batch['trg'].to(self.device)

                if self.model_name== 'transformer':
                    logit = self.model(src, trg[:, :-1])
                else:
                    logit = self.model(src, trg)
                
                loss = self.criterion(logit.contiguous().view(-1, self.output_dim), 
                                      trg[:, 1:].contiguous().view(-1)).item()

                greedy_pred = logit.argmax(-1).tolist()
                beam_pred = self.beam_search(src, trg)
                
                return loss, greedy_pred, beam_pred

                beam_bleu = self.get_bleu_score(beam_pred, trg)    
                greedy_bleu = self.get_bleu_score(greedy_pred, trg)

                tot_loss += loss
                tot_beam_bleu += beam_bleu
                tot_greedy_bleu += greedy_bleu

        tot_loss /= tot_len
        tot_beam_bleu /= tot_len
        tot_greedy_bleu /= tot_len
        
        print(f'Test Results on {self.model_name} model | Time: {self.measure_time(start_time, time.time())}')
        print(f">> Test Loss: {tot_loss:3f} | Test PPL: {math.exp(tot_loss):2f}")
        print(f">> Greedy BLEU: {tot_greedy_bleu:2f} | Beam BLEU: {tot_beam_bleu:2f}")