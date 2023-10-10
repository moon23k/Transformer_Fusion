## PLM Fusion
Recently, many pretrained language models have been showing excellent performance in various fields. 
However, in the case of most NLU-focused models led by BERT, they lack a decoder, making it challenging to intuitively apply them to NLG tasks that require an Encoder-Decoder structure. 
To mend this problem, this repository covers a series of experiments of incorporating a decoder to maximize the exceptional language understanding capabilities of pretrained language model in NLG tasks and evaluates each performance across three NLG tasks.

<br><br>

## Model Architecture 

| Simple Model | Fusion Model |
|--------------|-------------|
| <div align="center"><img src="https://github.com/moon23k/Transformer_Archs/assets/71929682/5d118ff7-7d8d-4093-ba73-e131e703f467" height="300px"></div> | <div align="center"><img src="https://github.com/moon23k/PLM_Fusion/assets/71929682/c685655b-b4c3-4007-89b8-48eb8b538ae3" width="400px"></div> |
| As the name suggests, the BERT Simple Seq2Seq Model is literally the simplest structural model to apply BERT to the NLG Task. This model follows Transformer architecture, but the only difference is that it uses BERT as an Encoder. | BERT Fused Seq2Seq Model follows the methodology that suggested on **`Incorporating BERT into Neural Machine Translation`** paper. The model fuses BERT's Last Hidden States Vector via Self Attention mechanism on its own Encoder & Decoder. |

<br><br>

## Results

| Model Type   | Machine Translation | Dialogue Generation | Summarization |
| :---:        | :---: | :---: | :---: |
| Simple Model | - | - | - |
| Fusion Model | - | - | - |

<br><br>

## Reference
* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
* [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)
* [Incorporating BERT into Neural Machine Translation](https://arxiv.org/abs/2002.06823)
* [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461)
  
