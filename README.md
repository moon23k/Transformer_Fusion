## NMT_BERT

Fine-Tuning Large-Scale Pre-Trained model represented by BERT to a specific task is effectively applied to various NLP tasks.
However, most of the applied tasks are simple tasks which only add a top layer to the BERT, relatively are not applied well to the Generation Task on the Encoder-Decoder structure. Pointing out this problem, a paper named **'Incorporating BERT into Neural Machine Translation'** suggests a methodology of using BERT to NMT task. This repo, referring to the aforementioned paper, implements how to apply BERT into NMT Task. And then compares the performance difference with that of the Transformer.

<br>
<br>

## Models

**Base Line**

Vanilla Transformer Architecture for NMT Task is set as Base Line model in this experiment. 
In order to compare only on the models, other Variables are set to be the same as possible.

Use of BERT Tokenizer from Hugging Face.

<br>

**Simple Fine-Tune**

This model simplest way to use BERT in downstream Task.
Use BERT as Encoder and connect it with Transformer Decoder.

To reduce Discripency between Pre-Trained Encoder and From-Scratch Decoder, Optimization goes by 2 different ways. 

<br>

**BERT Fused**

Use BERT as context Embedding Module and Fuse output via

<br>
<br>

## Configurations

<br>
<br>

## Result

<br>
<br>

## Reference
* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/2002.06823)
* [Incorporating BERT into Neural Machine Translation](https://arxiv.org/abs/1810.04805)
* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
