# Getting More from Less: Large Language Models are Good Spontaneous Multilingual Learners

<p align="center">
  <a href="https://arxiv.org/abs/2405.13816"> 📃 Paper</a> |  
  <a href="https://shimao-zhang.github.io/"> 📭 Contact</a> 
</p>

&nbsp;
**Accepted by EMNLP 2024 (Oral Presentation).**

### :mountain: Overview

This repository shares the code and models of our latest work "Getting More from Less: Large Language Models are Good Spontaneous Multilingual Learners". In this work, we first discover and comprehensively investigate the spontaneous multilingual alignment improvement of LLMs. We find that LLMs instruction-tuned on the question translation data (i.e. without annotated answers) are able to encourage the alignment between English and a wide range of languages, even including those unseen during instruction-tuning. Additionally, we utilize different settings and mechanistic interpretability methods to analyze the LLM's performance in the multilingual scenario comprehensively. Our work suggests LLMs' enormous potential for improving multilingual alignment efficiently with great language generalization and task generalization.

### :chart_with_upwards_trend: Benchmarks & Datasets

We provide the benchmarks and datasets we utilize in our experiments in the `./data`. We report the information in detail as below:

|                           Dataset                            |             Usage              |                          Languages                           |       Path        |
| :----------------------------------------------------------: | :----------------------------: | :----------------------------------------------------------: | :---------------: |
| [Amazon Reviews Polarity](https://huggingface.co/datasets/amazon_polarity/viewer/amazon_polarity/train) | Question Translation Alignment |                              \                               | ./data/ap_emotion |
| [SNLI](https://huggingface.co/datasets/stanfordnlp/snli/viewer/plain_text/train) | Question Translation Alignment |                              \                               |    ./data/snli    |
| [PAWS](https://huggingface.co/datasets/google-research-datasets/paws/viewer/labeled_final/train) | Question Translation Alignment |                              \                               |    ./data/paws    |
| [Amazon Reviews Polarity](https://huggingface.co/datasets/amazon_polarity/viewer/amazon_polarity/test) |           Evaluation           | en, zh, de, fr, es, it, nl, ja, ru, sv, sl, pl, bg, no, ms, is, hi, th, sw, bn | ./data/ap_emotion |
| [SNLI](https://huggingface.co/datasets/stanfordnlp/snli/viewer/plain_text/test) |           Evaluation           | en, zh, de, fr, es, it, nl, ja, ru, sv, sl, pl, bg, no, ms, is, hi, th, sw, bn |    ./data/snli    |
| [PAWS](https://huggingface.co/datasets/google-research-datasets/paws/viewer/labeled_final/test) |           Evaluation           | en, zh, de, fr, es, it, nl, ja, ru, sv, sl, pl, bg, no, ms, is, hi, th, sw, bn |    ./data/paws    |

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

For finetuning, you can use the hyperparameters below:

```bash
#!/bin/bash

export HF_HOME=/home/huggingface_cache_path

CUDA_VISIBLE_DEVICES=0 python ./src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path model_name_or_path \
    --dataset dataset_name \
    --dataset_dir ./data \
    --template template_name \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir output_dir_path \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps total_step/10 \
    --save_steps 150000 \
    --eval_steps 50 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --val_size 0.05 \
    --plot_loss \
    --fp16
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

* evaluating with PAWS

```bash
cd ./scripts
bash run_paws_eval.bash
```

* logit lens

```bash
cd ./scripts
bash run_emotion.bash
```

* Principal Component Analysis

  Running the Jupyter file `knowledge_finding.ipynb`

### :evergreen_tree: Citation

If you find this repository helpful, feel free to cite our paper.

You can just follow the citation information of ACL Anthology or Google Scholar:

```
@inproceedings{zhang-etal-2024-getting,
    title = "Getting More from Less: Large Language Models are Good Spontaneous Multilingual Learners",
    author = "Zhang, Shimao  and
      Gao, Changjiang  and
      Zhu, Wenhao  and
      Chen, Jiajun  and
      Huang, Xin  and
      Han, Xue  and
      Feng, Junlan  and
      Deng, Chao  and
      Huang, Shujian",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.457",
    pages = "8037--8051",
    abstract = "Recently, Large Language Models (LLMs) have shown impressive language capabilities, while most of them have very unbalanced performance across different languages. Multilingual alignment based on the translation parallel data is an effective method to enhance LLMs{'} multilingual capabilities. In this work, we first discover and comprehensively investigate the spontaneous multilingual alignment of LLMs. Firstly, we find that LLMs instruction-tuned on the question translation data (i.e. without annotated answers) are able to encourage the alignment between English and a wide range of languages, even including those unseen during instruction-tuning. Additionally, we utilize different settings and mechanistic interpretability methods to analyze the LLM{'}s performance in the multilingual scenario comprehensively. Our work suggests that LLMs have enormous potential for improving multilingual alignment efficiently with great language generalization and task generalization.",
}
```