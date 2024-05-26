# Large Language Models are Good Spontaneous Multilingual Learners:  Is the Multilingual Annotated Data Necessary?

<p align="center">
  <a href="https://arxiv.org/abs/2405.13816"> 📃 Paper</a> |  
  <a href="https://shimao-zhang.github.io/"> 📭 Contact</a> 
</p>


### :mountain: Overview

This repository shares the code and models of our latest work "Large Language Models are Good Spontaneous Multilingual Learners: Is the Multilingual Annotated Data Necessary?". In this work, we find that only tuning on questions (without annotated answers) in a small number of languages can bring significant multilingual improvements even across a wide range of languages unseen during the instruction-tuning process. Additionally, we utilize different settings and mechanistic interpretability methods to comprehensively analyze the LLM's performance in the multilingual scenario.

### :chart_with_upwards_trend: Benchmarks & Datasets

We provide the benchmarks and datasets we utilize in our experiments in the `./data`. We report the information in detail as below:

|                           Dataset                            |             Usage              |                          Languages                           |       Path        |
| :----------------------------------------------------------: | :----------------------------: | :----------------------------------------------------------: | :---------------: |
| [Amazon Reviews Polarity](https://huggingface.co/datasets/amazon_polarity/viewer/amazon_polarity/train) | Question Translation Alignment |                              \                               | ./data/ap_emotion |
| [SNLI](https://huggingface.co/datasets/stanfordnlp/snli/viewer/plain_text/train) | Question Translation Alignment |                              \                               |    ./data/snli    |
| [Amazon Reviews Polarity](https://huggingface.co/datasets/amazon_polarity/viewer/amazon_polarity/test) |           Evaluation           | en, zh, de, fr, es, it, nl, ja, ru, sv, sl, pl, bg, no, ms, is, hi, th, sw, bn | ./data/ap_emotion |
| [SNLI](https://huggingface.co/datasets/stanfordnlp/snli/viewer/plain_text/test) |           Evaluation           | en, zh, de, fr, es, it, nl, ja, ru, sv, sl, pl, bg, no, ms, is, hi, th, sw, bn |    ./data/snli    |

### :jigsaw: Installation

To install this repository, follow these steps:

```bash
git clone https://github.com/Shimao-Zhang/LLM-Multilingual-Learner.git
cd LLM-Multilingual-Learner
pip install -r requirements.txt
```

### :hammer_and_wrench: Training

We train our models based on the [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory). 

You should replace the path of the model and data in `./LLaMA-Factory/sft_question_single_lora.bash` with the appropriate paths, and you should also use the corresponding template.

* finetuning

```bash
bash ./LLaMA-Factory/sft_question_single_lora.bash
```

* merge

```bash
bash ./LLaMA-Factory/merge_lora_weights.bash
```

### :straight_ruler: Evaluation & Analysis

We evaluate the model by constrained decoding and calculating the accuracy. To evaluate the model performance, you can use the following command. Note that before running the scripts, you should set the appropriate model_size, target_lang, and model path in the corresponding `.py` file.

* evaluating with Amazon Reviews Polarity

```bash
cd ./scripts
bash run_emotion_eval.bash
```

* evaluating with SNLI

```bash
cd ./scripts
bash run_snli_eval.bash
```

* logit lens

```bash
cd ./scripts
bash run_emotion.bash
```

* Principal Component Analysis
  Running the Jupyter file `knowledge_finding.ipynb`

### :evergreen_tree: Citation

If you find this repository helpful, feel free to cite our paper. The following citation information is obtained from Google Scholar:

```
@article{zhang2024large,
  title={Large Language Models are Good Spontaneous Multilingual Learners: Is the Multilingual Annotated Data Necessary?},
  author={Zhang, Shimao and Gao, Changjiang and Zhu, Wenhao and Chen, Jiajun and Huang, Xin and Han, Xue and Feng, Junlan and Deng, Chao and Huang, Shujian},
  journal={arXiv preprint arXiv:2405.13816},
  year={2024}
}
```