## NMT Back Translation

A large amount of well-formed data is essential to improve deep learning model performace. 
However, getting a lot of data in all situations is difficult.
One methodology to overcome data scarcity and drive performance improvement is **back translation**.
This repo covers various back translation methodologies and compares performance of each method.
For accurate comparison, other variables except for back translation methodologies are fixed.

<br>
<br>

## Methodologies

**Amount of Synthetic Data Tuning**


<br>

**How to Generate Synthetic Data Generating**
As for how to create sentences, there are typically options such as greedy, beam, top-p sampling.
Beam and Greedy use the mae to generate sequence, so they tend to produce more precise sentences.
On the other hand, top-p sampling has more possibilities for generating sentences compared to the previous two methods, but it is highly likely to generate sentences that are not sophisticated.

In this experiment, Greedy Search is used to account for the trade off between generation speed and accuracy.

<br>

**Corrupting Synthetic Data**

One of the keys to boosting performance through back translation is inducing more difficult problems to be solved with low quality data.
Just like the way Language model use Auto Encoding via intentionally corrupted sentence, we also corrupt Synthetic Data to improve Translation Model Performance.

* Masking
* Delete Random Token


<br>
<br>

## Experimental Setups

The experiment was conducted based on the Korean-English daily conversation dataset provided by aihub.
And two pretrained kobart models were used, each of **circulus/kobart-trans-ko-en-v2** and **circulus/kobart-trans-en-ko-v2**.


| &emsp; **Dataset desc**                              | &emsp; **Model Config**                 | &emsp; **Training Config**               |
| :---                                                 | :---                                    | :---                                     |
| **`Data from:`** &hairsp; `AI Hub`                   | **`Input Dimension:`** `30,000`         | **`Epochs:`** `10`                       |
| **`Specific Datasets:`** &hairsp; `Dialogue, Daily`  | **`Output Dimension:`** `30,000`        | **`Batch Size:`** `32`                   |
| **`Total Dataset Volumn:`** &hairsp; `50,000` &emsp; | **`Embedding Dimension:`** `256` &emsp; | **`Learning Rate:`** `5e-4`              |
| **`Train Dataset Volumn:`** &hairsp; `48,000`        | **`Hidden Dimension:`** `512`           | **`iters_to_accumulate:`** `4`           |
| **`Valid  Dataset Volumn:`** &hairsp; `1,000`        | **`N Layers:`** `2`                     | **`Gradient Clip Max Norm:`** `1` &emsp; |
| **`Test  Dataset Volumn:`** &hairsp; `1,000`         | **`Drop-out Ratio:`** `0.5`             | **`Apply AMP:`** `True`                  |


<br>
<br>

## Results

| Training Method | Lang Pair | Training Data Volumn | Best Training Loss | BLEU Score |
| :---: | :---: | :---: | :---: | :---: |
| Vanilla           | Ko-En | 48,000 | - | 34.41 |
| Vanilla           | En-Ko | 48,000 | - | 11.71 |
| Back Translation  | Ko-En | - | - | - |
| Back Translation  | En-Ko | - | - | - |
| Back + Corruption | Ko-En | - | - | - |
| Back + Corruption | En-Ko | - | - | - |

<br>
<br>


## Reference
* [Understanding Back-Translation at Scale](https://arxiv.org/abs/1808.09381)
