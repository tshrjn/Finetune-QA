# SQUAD Fine-Tuning
BERT, RoBERTa fine-tuning over SQuAD Dataset using pytorch-lighting, transformers & nlp.

Example Usage:
`python main.py --gpus 1, --workers 20 --bs 5 --max_epochs 10`

# Features
* Pytorch-Lightning Goodies
    * All args from the trainer
    * Multi-GPU support
* Huggingface Tramsformer easy plug-n-play
* Huggingface NLP Dataset easy data handling
