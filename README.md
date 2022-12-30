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

* Greedy

* Beam Search

* Top-p Sampling

<br>

**Corrupting Synthetic Data**

One of the keys to boosting performance through back translation is inducing more difficult problems to be solved with low quality data.
Just like the way Language model use Auto Encoding via intentionally corrupted sentence, we also corrupt Synthetic Data to improve Translation Model Performance.

* Masking
* Delete Random Token


<br>
<br>

## Results

<br>
<br>

## Result

<br>
<br>

## Reference
* [Understanding Back-Translation at Scale](https://arxiv.org/abs/1808.09381)
