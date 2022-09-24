import torch, operator
import torch.nn.functional as F
from queue import PriorityQueue
from collections import namedtuple


class Search:
    def __init__(self, config, model):
        self.beam_size = 4
        self.device = config.device
        self.bos_idx = config.bos_idx
        self.eos_idx = config.eos_idx
        self.pad_idx = config.pad_idx
        self.output_dim = config.output_dim
        self.model_name = config.model_name
        
        self.model = model
        self.model.eval()

        if self.model_name == 'seq2seq':
            self.Node = namedtuple('Node', ['prev_node', 'hidden', 'log_prob', 'pred', 'preds', 'length'])
            
        elif self.model_name == 'transformer':
            self.Node = namedtuple('Node', ['prev_node', 'pred', 'pred_mask', 'log_prob', 'length'])


    def beam_score(self, node, no_repeat_ngram_size, min_length=5, alpha=1.2):
        overlap = max([node.pred.tolist().count(token) for token in node.pred.tolist() if token != self.pad_idx])

        if overlap > no_repeat_ngram_size:
            overlap_penalty = -1
        else:
            overlap_penalty = 1
        
        length_penalty = ((node.length + min_length) / (1 + min_length)) ** alpha
        score = node.log_prob / length_penalty
        score = score * overlap_penalty
        return score


    def _encode(self, src):
        if len(list(src.size())) == 1:
            src = src.unsqueeze(0)

        if self.model_name == 'seq2seq':
            hiddens = self.model.encoder(src)
            return hiddens
        elif self.model_name == 'transformer':
            src_masks = self.model.pad_mask(src)
            memories = self.model.encoder(src, src_masks)
            return memories, src_masks


    def _decode(self, curr_node, memory=None, src_mask=None):
        if self.model_name == 'seq2seq':
            out, _hidden = self.model.decoder(curr_node.pred, curr_node.hidden)
            
            logits, preds = torch.topk(out, self.beam_size)
            log_probs = F.log_softmax(logits, dim=-1)
            return logits, preds, log_probs, _hidden

        elif self.model_name == 'transformer':
            dec_out = self.model.decoder(curr_node.pred.unsqueeze(0), memory, src_mask, curr_node.pred_mask)
            out = self.model.fc_out(dec_out)[:, -1, :]
            logits, preds = torch.topk(out, self.beam_size)
            logits, preds = logits.squeeze(1), preds.squeeze(1)
            log_probs = F.log_softmax(logits, dim=-1)
            return logits, preds, log_probs


    def select_vector(self, idx, hiddens=None, memories=None, src_masks=None):
        if self.model_name == 'seq2seq':
            hidden = (hiddens[0][:, idx, :].unsqueeze(1).contiguous(),
                      hiddens[1][:, idx, :].unsqueeze(1).contiguous())
            return hidden

        elif self.model_name == 'attention':
            hidden = hiddens[:, idx, :].unsqueeze(1).contiguous()
            return hidden
        
        elif self.model_name == 'transformer':
            memory = memories[idx, :, :].unsqueeze(0).contiguous()
            src_mask = src_masks[idx, :, :].unsqueeze(0).contiguous()
            return memory, src_mask


    def add_nodes(self, curr_node, logits, preds, log_probs, no_repeat_ngram_size, _hidden=None):
        Node = self.Node
        if self.model_name == 'seq2seq':
            for k in range(self.beam_size):
                pred = preds[0][k].view(1)
                log_prob = log_probs[0][k].item()

                next_node = Node(prev_node = curr_node,
                                 hidden = (_hidden[0].contiguous(), _hidden[1].contiguous()),
                                 log_prob = curr_node.log_prob + log_prob,
                                 pred = pred.contiguous(),
                                 preds = curr_node.preds + [pred.item()],
                                 length = curr_node.length + 1)
                next_score = self.beam_score(next_node, no_repeat_ngram_size)
                nodes.put((next_score, next_node))

        elif self.model_name == 'transformer':
            for k in range(self.beam_size):
                pred = preds[0][k].view(1)
                log_prob = log_probs[0][k].item()
                new_pred = torch.cat([curr_node.pred, pred])
                new_pred_mask = self.model.dec_mask(new_pred.contiguous().unsqueeze(0))           
                
                next_node = Node(prev_node = curr_node,
                                 pred = new_pred,
                                 pred_mask = new_pred_mask,
                                 log_prob = curr_node.log_prob + log_prob,
                                 length = curr_node.length + 1)
                next_score = self.beam_score(next_node, no_repeat_ngram_size)                
                nodes.put((next_score, next_node))


    def get_output(self, top_node):
        if self.model_name == 'seq2seq':
            output = []
            while top_node.prev_node is not None:
                output.append(top_node.pred.item())
                top_node = top_node.prev_node
            return output[::-1]

        elif self.model_name == 'transformer':
            return top_node.pred.tolist()


    def beam_search(self, src, trg, topk=1):
        outputs = []
        batch_size, max_len = trg.shape
        start_tensor = torch.LongTensor([self.bos_idx]).to(self.device)
        overlaps = [max([seq.count(token) for token in seq if token != self.pad_idx]) for seq in src.tolist()]

        if self.model_name == 'seq2seq':
            hiddens = self._encode(src)
            Node = self.Node
            
        elif self.model_name == 'transformer':
            memories, src_masks = self._encode(src)
            Node = self.Node
        
        for idx in range(batch_size):
            if self.model_name == 'seq2seq':
                hidden = self.select_vector(idx=idx, hiddens=hiddens)
                start_node = Node(prev_node = None, 
                                  hidden = hidden, 
                                  log_prob = 0.0, 
                                  pred = start_tensor, 
                                  preds = [start_tensor.item()],
                                  length = 0)
            elif self.model_name == 'transformer':
                memory, src_mask = self.select_vector(idx=idx, memories=memories, src_masks=src_masks)
                start_node = Node(prev_node = None,
                                  pred = start_tensor,
                                  pred_mask = self.model.dec_mask(start_tensor),
                                  log_prob = 0.0,
                                  length = 0)

            global nodes
            nodes = PriorityQueue()
            for _ in range(self.beam_size):
                nodes.put((0, start_node))

            end_nodes, top_nodes = [], []
            for t in range(max_len):
                curr_nodes = [nodes.get() for _ in range(self.beam_size)]
                for curr_score, curr_node in curr_nodes:
                    #set different conditions depending on the models
                    if self.model_name == 'seq2seq':
                        condition = curr_node.pred.item() == self.eos_idx and curr_node.prev_node != None
                    elif self.model_name == 'transformer':
                        condition = curr_node.pred[-1].item() == self.eos_idx and curr_node.prev_node != None 
                    
                    if condition:
                        if curr_node.length <= 3:  #if generate length is too short, skip the node
                            continue
                        end_nodes.append((curr_score, curr_node))
                        continue
                        
                    if self.model_name == 'seq2seq':
                        logits, preds, log_probs, _hidden = self._decode(curr_node)
                        self.add_nodes(curr_node, logits, preds, log_probs, overlaps[idx], _hidden)
                    elif self.model_name == 'transformer':
                        logits, preds, log_probs = self._decode(curr_node, memory, src_mask)
                        self.add_nodes(curr_node, logits, preds, log_probs, overlaps[idx])

                    if not t:
                        break

            if len(end_nodes) == 0:
                _, top_node = nodes.get()
            else:
                _, top_node = sorted(end_nodes, key=operator.itemgetter(0), reverse=True)[0]
            
            output = self.get_output(top_node)
            outputs.append(output)
        return outputs
