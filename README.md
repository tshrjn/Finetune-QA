# SQUAD Fine-Tuning
BERT, RoBERTa fine-tuning over SQuAD Dataset using pytorch-lighting, transformers & nlp.

### Usage

Example Usage:
`python main.py --gpus 1, --qa_model distilroberta-base --workers 20 --bs 5 --max_epochs 10`

A Useful WANDB environment variables:
```
WANDB_MODE=dryrun
WANDB_ENTITY=nlp
```

### Install

```bash
pip install -r requirements.txt
```

### Features
* ⚡️Pytorch-Lightning: Goodies
    * All `Trainer` flags as args
    * Multi-GPU support
* 🤗 Transformer: easy plug-n-play
* 🤗 NLP Dataset: easy data handling

### TODO:
TBD