import os, torch
from model import FusedModel, SimpleModel
from transformers import (
    BertGenerationDecoder,
    BertGenerationEncoder,
    EncoderDecoderModel
)



def count_params(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params

    
def check_size(model):
    param_size, buffer_size = 0, 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb



def load_model(config):
    if config.model_type == 'simple':
        model = SimpleModel(config)

    elif config.model_type == 'fused':
        model = FusedModel(config)
    
    elif config.model_type == 'enc_dec':
        
        encoder = BertGenerationEncoder.from_pretrained(
            config.bert_mname, 
            bos_token_id=config.bos_id,
            eos_token_id=config.eos_id
        )

        decoder = BertGenerationDecoder.from_pretrained(
            config.bert_mname, 
            add_cross_attention=True, 
            is_decoder=True,
            bos_token_id=config.bos_id,
            eos_token_id=config.eos_id
        )

        model = EncoderDecoderModel(encoder=encoder, decoder=decoder)        

        model.config.encoder.decoder_start_token_id = config.bos_id
        model.config.decoder.decoder_start_token_id = config.bos_id
        model.config.decoder_start_token_id = config.bos_id
        
        model.config.pad_token_id = config.pad_id
        model.config.vocab_size = config.vocab_size

    print(f"BERT {config.model_type.upper()} Model for has loaded")

    
    if config.mode != 'train':
        assert os.path.exists(config.ckpt)
        model_state = torch.load(config.ckpt, map_location=config.device)['model_state_dict']
        model.load_state_dict(model_state)
        print(f"Trained Model States has loaded on the Model")

    
    print(f"--- Model Params: {count_params(model):,}")
    print(f"--- Model  Size : {check_size(model):.3f} MB\n")
    
    return model.to(config.device)