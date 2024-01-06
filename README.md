## Transformer Fusion
Recently, many pretrained language models have been showing excellent performance in various fields. 
However, in the case of most NLU-focused models led by BERT, they lack a decoder, making it challenging to intuitively apply them to NLG tasks that require an Encoder-Decoder structure. 
To mend this problem, this repository covers a series of experiments of incorporating a decoder to maximize the exceptional language understanding capabilities of pretrained language model in NLG tasks and evaluates each performance across three NLG tasks.

<br><br>

## Architecture 

| Simple Model | Fusion Model |
|--------------|-------------|
| <div align="center"><img src="https://github.com/moon23k/Transformer_Archs/assets/71929682/5d118ff7-7d8d-4093-ba73-e131e703f467" height="300px"></div> | <div align="center"><img src="https://github.com/moon23k/PLM_Fusion/assets/71929682/c685655b-b4c3-4007-89b8-48eb8b538ae3" width="400px"></div> |
| As the name suggests, the BERT Simple Seq2Seq Model is literally the simplest structural model to apply BERT to the NLG Task. This model follows Transformer architecture, but the only difference is that it uses BERT as an Encoder. | BERT Fused Seq2Seq Model follows the methodology that suggested on **`Incorporating BERT into Neural Machine Translation`** paper. The model fuses BERT's Last Hidden States Vector via Self Attention mechanism on its own Encoder & Decoder. |

<br><br>

## Setup
The default values for experimental variables are set as follows, and each value can be modified by editing the config.yaml file. <br>

| **Tokenizer Setup**                         | **Model Setup**                   | **Training Setup**                |
| :---                                        | :---                              | :---                              |
| **`Tokenizer Type:`** &hairsp; `BPE`        | **`Input Dimension:`** `15,000`   | **`Epochs:`** `10`                |
| **`Vocab Size:`** &hairsp; `15,000`         | **`Output Dimension:`** `15,000`  | **`Batch Size:`** `32`            |
| **`PAD Idx, Token:`** &hairsp; `0`, `[PAD]` | **`Hidden Dimension:`** `256`     | **`Learning Rate:`** `5e-4`       |
| **`UNK Idx, Token:`** &hairsp; `1`, `[UNK]` | **`PFF Dimension:`** `512`        | **`iters_to_accumulate:`** `4`    |
| **`BOS Idx, Token:`** &hairsp; `2`, `[BOS]` | **`Num Layers:`** `3`             | **`Gradient Clip Max Norm:`** `1` |
| **`EOS Idx, Token:`** &hairsp; `3`, `[EOS]` | **`Num Heads:`** `8`              | **`Apply AMP:`** `True`           |

<br>To shorten the training speed, techiques below are used. <br> 
* **Accumulative Loss Update**, as shown in the table above, accumulative frequency has set 4. <br>
* **Application of AMP**, which enables to convert float32 type vector into float16 type vector.

<br><br>

## How to Use
```
├── ckpt                    --this dir saves model checkpoints and training logs
├── config.yaml             --this file is for setting up arguments for model, training, and tokenizer 
├── data                    --this dir is for saving Training, Validataion and Test Datasets
├── model                   --this dir contains files for Deep Learning Model
│   ├── common.py
│   ├── fusion.py
│   ├── __init__.py
│   └── simple.py
├── module                  --this dir contains a series of modules
│   ├── data.py
│   ├── generate.py
│   ├── __init__.py
│   ├── model.py
│   ├── test.py
│   └── train.py
├── README.md
├── run.py                 --this file includes codes for actual tasks such as training, testing, and inference to carry out the practical aspects of the work
└── setup.py               --this file contains a series of codes for preprocessing data, training a tokenizer, and saving the dataset

```

**First clone git repo in your local env**
```
git clone https://github.com/moon23k/LSTM_Anchors
```

<br>

**Download and Process Dataset via setup.py**
```
bash setup.py -task [all, translation, dialogue, summarization]
```

<br>

**Execute the run file on your purpose (search is optional)**
```
python3 run.py -task [translation, dialogue, summarization] \
               -mode [train, test, inference] \
               -model [simple, fusion] \
               -search [greedy, beam]
```


<br><br>

## Reference
* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
* [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)
* [Incorporating BERT into Neural Machine Translation](https://arxiv.org/abs/2002.06823)
* [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461)
  
