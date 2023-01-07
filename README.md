## NMT BERT Fused

BERT is an excellent model that shows good performance on various NLU tasks. However, since it consists only of an encoder, it has a disadvantage that it is difficult to use it to generate sentences. To solve this problem and use bert on Natural Language Generation Tasks, **Incorporating BERT into Neural Machine Translation** paper present the **BERT Fused Model Architecture**.
This repo implemnets BERT Fused Model pytorch code, and compare it with **BERT Generation Model**.

<br>
<br>

## Model desc

* BERT Fused Model

<br>

* BERT Generation Model


<br>
<br>

## Experimental Setups


| &emsp; **Dataset desc**                              | &emsp; **Model Config**                 | &emsp; **Training Config**               |
| :---                                                 | :---                                    | :---                                     |
| **`Dataset:`** &hairsp; `WMT14 En-De`                | **`Input Dimension:`** `30,000`         | **`Epochs:`** `10`                       |
| -                                                    | **`Output Dimension:`** `30,000`        | **`Batch Size:`** `32`                   |
| **`Total Dataset Volumn:`** &hairsp; `50,000` &emsp; | **`Embedding Dimension:`** `256` &emsp; | **`Learning Rate:`** `5e-4`              |
| **`Train Dataset Volumn:`** &hairsp; `48,000`        | **`Hidden Dimension:`** `512`           | **`iters_to_accumulate:`** `4`           |
| **`Valid  Dataset Volumn:`** &hairsp; `1,000`        | **`N Layers:`** `2`                     | **`Gradient Clip Max Norm:`** `1` &emsp; |
| **`Test  Dataset Volumn:`** &hairsp; `1,000`         | **`Drop-out Ratio:`** `0.5`             | **`Apply AMP:`** `True`                  |


<br>
<br>

## Results

| Model | Lang Pair | Best Training Loss | BLEU Score |
| :---: | :---: | :---: | :---: |
| BERT Fused           | En-De | - | 34.41 |
| BERT Fused        | De-En | - | 11.71 |
| BERT Generation  | En-De | - | - |
| BERT Generation  | De-En | - | - |

<br>
<br>


## Reference
* [Incorporating BERT into Neural Machine Translation](https://arxiv.org/abs/2002.06823)
