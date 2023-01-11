## NMT BERT Fused

The BERT is a pretrained language model, which consists only of Transformer's encoder layers. By Pre-Training from large datasets, the BERT develops its ability to understand languages. BERT has proved its power, showing sota performances on various NLU tasks. However it is difficult to use it to generation task, since BERT only consists of encoder layers. 
To mend this problem, there has been a series of researches to use BERT well on generative tasks.
**`Incorporating BERT into Neural Machine Translation`** and **`Leveraging Pre-trained Checkpoints for Sequence Generation Tasks`** are representative studies.
This repo implemnets three different methods to use BERT on NLG Tasks and compare the performance of each on Machine Translation Task.

<br>
<br>

## Model desc 
**BERT Simple**
> As the name suggests, the simple model is literally the simplest structural model for applying BERT to the NLG Task. This model follows Transformer architecture, but the only difference is that it uses BERT as an Encoder. 

<br>

**BERT Fused**


<br>

**BERT Generation Pre Trained Model**


<br>
<br>

## Experimental Setups in Common


| &emsp; **Dataset desc**                              | &emsp; **Model Config**                 | &emsp; **Training Config**               |
| :---                                                 | :---                                    | :---                                     |
| **`Dataset:`** &hairsp; `WMT14 En-De`                | **`Input Dimension:`** `30,000`         | **`Epochs:`** `10`                       |
| -                                                    | **`Output Dimension:`** `30,000`        | **`Batch Size:`** `32`                   |
| **`Total Dataset Volumn:`** &hairsp; `36,000` &emsp; | **`Embedding Dimension:`** `256` &emsp; | **`Learning Rate:`** `5e-4`              |
| **`Train Dataset Volumn:`** &hairsp; `30,000`        | **`Hidden Dimension:`** `512`           | **`iters_to_accumulate:`** `4`           |
| **`Valid  Dataset Volumn:`** &hairsp; `3,000`        | **`N Layers:`** `2`                     | **`Gradient Clip Max Norm:`** `1` &emsp; |
| **`Test  Dataset Volumn:`** &hairsp; `3,000`         | **`Drop-out Ratio:`** `0.1`             | **`Apply AMP:`** `True`                  |


<br>
<br>

## Results

| Model | Lang Pair | Best Validation Loss | BLEU Score |
| :---: | :---: | :---: | :---: |
| BERT Simple     | En-De | - | - |
| BERT Simple     | De-En | - | - |
| BERT Fused      | En-De | - | 34.41 |
| BERT Fused      | De-En | - | 11.71 |
| BERT Generation | En-De | - | - |
| BERT Generation | De-En | - | - |

<br>
<br>


## Reference
* [Incorporating BERT into Neural Machine Translation](https://arxiv.org/abs/2002.06823)
* [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461)
