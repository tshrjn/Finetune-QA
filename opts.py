import argparse
import multiprocessing as mp
from pytorch_lightning import Trainer

def get_args():
    parser = argparse.ArgumentParser(description='QA')
    parser = Trainer.add_argparse_args(parser)  # Adds all pl's trainer args (like max_epochs)

    optim_args = parser.add_argument_group('Optimization related arguments')
    optim_args.add_argument('--bs', type=int,  default=512, help='Batch Size')
    optim_args.add_argument('--lr', type=float,  default=1e-4, help='Initial Learning rate')

    data_args = parser.add_argument_group('Data related arguments')
    data_args.add_argument('--percent', type=int, default=100, help='Data% to train')

    model_args = parser.add_argument_group('Model related arguments')
    model_args.add_argument('--qa_model', type=str, default='deepset/roberta-base-squad2',
                            help='Model name')
    # Model Choices:
    # QA layer pre-finetuned: https://huggingface.co/models?filter=pytorch,question-answering&search=bert
    # Commonly used: twmkn9/distilroberta-base-squad2, deepset/roberta-base-squad2
    # Finetune QA layer from scratch: roberta-base, roberta-large, distilroberta-base, distilroberta-base


    misc_args = parser.add_argument_group('Logging related & Misc arguments')
    misc_args.add_argument('--seed', type=int, default=777, help='Random Seed')
    misc_args.add_argument('-t','--tags', nargs='+', default=[],help='W&B Tags to associate with run')
    misc_args.add_argument('--workers', type=int, default=min(8, mp.cpu_count()-1), help='Number of parallel worker threads')


    args = parser.parse_args()
    return args
