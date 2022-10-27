import torch, operator
import torch.nn.functional as F
from queue import PriorityQueue
from collections import namedtuple




def get_score(node, max_repeats, pad_idx, min_length=5, alpha=1.2):
    pred_repeats = max([node.pred.tolist().count(token) for token in node.pred.tolist() if token != pad_idx])

    if pred_repeats > max_repeats + 5:
        repeat_penalty = -1
    else:
        repeat_penalty = 1
    
    length_penalty = ((node.length + min_length) / (1 + min_length)) ** alpha
    score = node.log_prob / length_penalty
    score = score * repeat_penalty
    return score


def get_output(top_node, model_name):
    if model_name != 'transformer':
        output = []
        while top_node.prev_node is not None:
            output.append(top_node.pred.item())
            top_node = top_node.prev_node
        return output[::-1]

    elif model_name == 'transformer':
        return top_node.pred.tolist()


def src_params(src, pad_idx, bos_idx):
    if src.dim == 1:
        src = src.unsqueeze(0)
    batch_size, max_len = src.size(0), src.size(1) + 50
    src_repeats = [max([seq.count(token) for token in seq if token != pad_idx]) for seq in src.tolist()]    
    return src, batch_size, max_len, src_repeats


def get_nodes(start_node, beam_size):
    nodes = PriorityQueue()
    for _ in range(beam_size):
        nodes.put((0, start_node))        
    return nodes, [], []    




class Search:
    def __init__(self, config, model):
        self.model = model
        self.beam_size = 4
        self.device = config.device
        self.bos_idx = config.bos_idx
        self.eos_idx = config.eos_idx
        self.pad_idx = config.pad_idx
        self.output_dim = config.output_dim
        self.model_name = config.model_name


    def beam_search(self, src, topk=1):
        outputs = []
        Node = namedtuple('Node', ['prev_node', 'pred', 'pred_mask', 'log_prob', 'length'])
        
        start_tensor = torch.LongTensor([self.bos_idx]).to(self.device)
        src, batch_size, max_len, src_repeats = src_params(src, self.pad_idx, self.bos_idx)

        src_masks = self.model.pad_mask(src)
        memories = self.model.encoder(src, src_masks)

        for idx in range(batch_size):
            memory = memories[idx, :, :].unsqueeze(0).contiguous()
            src_mask = src_masks[idx, :, :].unsqueeze(0).contiguous()
            
            start_node = Node(prev_node = None,
                              pred = start_tensor,
                              pred_mask = self.model.dec_mask(start_tensor),
                              log_prob = 0.0,
                              length = 0)

            nodes, end_nodes, top_nodes = get_nodes(start_node, self.beam_size)
            for t in range(max_len):
                curr_nodes = [nodes.get() for _ in range(self.beam_size)]
                for curr_score, curr_node in curr_nodes:
                    if curr_node.pred[-1].item() == self.eos_idx and curr_node.prev_node != None:
                        if curr_node.length <= 3:  #if generate length is too short, skip the node
                            continue
                        end_nodes.append((curr_score, curr_node))
                        continue
                        
                    dec_out = self.model.decoder(curr_node.pred.unsqueeze(0), memory, src_mask, curr_node.pred_mask)
                    out = self.model.fc_out(dec_out)[:, -1, :]
                    logits, preds = torch.topk(out, self.beam_size)
                    logits, preds = logits.squeeze(1), preds.squeeze(1)
                    log_probs = F.log_softmax(logits, dim=-1)

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
                        next_score = get_score(next_node, src_repeats[idx], self.pad_idx)                
                        nodes.put((next_score, next_node))
                    
                    if not t:
                        break

            if len(end_nodes) == 0:
                _, top_node = nodes.get()
            else:
                _, top_node = sorted(end_nodes, key=operator.itemgetter(0), reverse=True)[0]
            
            output = get_output(top_node, self.model_name)
            outputs.append(output)

        return outputs        


    def greedy_search(self, src):
        outputs = []
        src, batch_size, max_len, _ = src_params(src, self.pad_idx, self.bos_idx)

        e_mask, d_mask = self.model.pad_mask(src), self.model.dec_mask(trg)
        memory = self.model.encoder(src, e_mask)

        for idx in range(batch_size):
            for t in range(max_len):
                if pred == self.eos_id:
                    break

        return outputs 


    def greedy(config, model):
        for src, trg in zip(src_batch, trg_batch):
            max_len = src.size(-1) + 30
            src = src.unsqueeze(0)
            e_mask = model.pad_mask(src)
            
            with torch.no_grad():
                memory = model.encoder(src, e_mask)

            pred_seq = torch.Tensor([[config.bos_idx]]).long().to(config.device)
            for _ in range(max_len):
                d_mask = model.dec_mask(pred_seq)
                out = model.decoder(pred_seq, memory, e_mask, d_mask)
                out = F.softmax(model.fc_out(out), dim=-1)
                pred_tok = out.argmax(dim=-1, keepdim=True)[:, -1]

                if pred_tok.item() == config.eos_idx:
                    break
                pred_seq = torch.cat([pred_seq, pred_tok], dim=-1)

            _src = [tok for tok in src.squeeze().tolist() if tok != config.pad_idx]
            _trg = [tok for tok in trg.tolist() if tok != config.pad_idx]        