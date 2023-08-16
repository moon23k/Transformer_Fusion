## NMT BERT Fused

The BERT is a well known pretrained language model, which consists only of Transformer's encoder layers. By Pre-Training from large datasets, the BERT develops its ability to understand languages. BERT has improved its power, showing Sota performances on various NLU tasks. However it is difficult to use it to generation task, since BERT only consists of encoder layers. 
To mend this problem, there has been a series of researches to use BERT well on generative tasks.
**`Incorporating BERT into Neural Machine Translation`** and **`Leveraging Pre-trained Checkpoints for Sequence Generation Tasks`** are representative studies.
This repo implemnets three different methods to use BERT on NLG Tasks and compare the performance of each on Machine Translation Task.

<br>
<br>

## Model desc 
**BERT Simple Seq2Seq Model**
> As the name suggests, the BERT Simple Seq2Seq Model is literally the simplest structural model to apply BERT to the NLG Task. This model follows Transformer architecture, but the only difference is that it uses BERT as an Encoder.

<br>

**BERT Fused Seq2Seq Model**
> BERT Fused Seq2Seq Model follows the methodology that suggested on **`Incorporating BERT into Neural Machine Translation`** paper. The model fuses BERT's Last Hidden States Vector via Self Attention mechanism on its own Encoder & Decoder.

<br>

**BERT Generation Model**
> BERT Generation Model uses BERT both on Encoder & Decoder. The main idea of model structure borrowed from **`Leveraging Pre-trained Checkpoints for Sequence Generation Tasks`** paper and code implemetation based on Hugging Face's Transformers Library.

<br><br>

## Experimental Setups in Common


| &emsp; **Dataset Config**                            | &emsp; **Model Config**                 | &emsp; **Training Config**               |
| :---                                                 | :---                                    | :---                                     |
| **`Dataset:`** &hairsp; `WMT14 En-De`                | **`Input Dimension:`** `30,000`         | **`Epochs:`** `10`                       |
| **`Total Dataset Volumn:`** &hairsp; `36,000` &emsp; | **`Output Dimension:`** `30,000`        | **`Batch Size:`** `32`                   |
| **`Train Dataset Volumn:`** &hairsp; `30,000`        | **`Embedding Dimension:`** `256` &emsp; | **`Learning Rate:`** `5e-4`              |
| **`Valid  Dataset Volumn:`** &hairsp; `3,000`        | **`Hidden Dimension:`** `512`           | **`iters_to_accumulate:`** `4`           |
| **`Test  Dataset Volumn:`** &hairsp; `3,000`         | **`N Layers:`** `2`                     | **`Gradient Clip Max Norm:`** `1` &emsp; |
|                                                      | **`Drop-out Ratio:`** `0.1`             | **`Apply AMP:`** `True`                  |


<br>
<br>

## Results

| Model | Best Validation Loss | BLEU Score |
| :---: | :---: | :---: |
| BERT Simple     | - | - |
| BERT Fused      | - | 34.41 |
| BERT Generation | - | - |

<br>
<br>


## Reference
* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
* [Incorporating BERT into Neural Machine Translation](https://arxiv.org/abs/2002.06823)
* [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461)
