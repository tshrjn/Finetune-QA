import nlp
import torch
import transformers as tfs


tokenizer = None
def prepare_data(args):
    #Temporary fix for pl issue #2036
    global tokenizer
    tokenizer = tfs.AutoTokenizer.from_pretrained(args.qa_model, use_fast=True)
    print(tokenizer)

    def _prepare_ds(split):
        ds = nlp.load_dataset('squad',
            split=f'{split}[:{args.bs if args.fast_dev_run else f"{args.percent}%"}]')
        # ds.cleanup_cache_files()
        ds = ds.map(convert_to_features, batched=True, batch_size=args.bs)

        columns_to_return = ['input_ids', 'token_type_ids', 'attention_mask',
                            'start_positions', 'end_positions']
        ds.set_format(type='torch', columns=columns_to_return)
        dl = torch.utils.data.DataLoader(ds, batch_size=args.bs, num_workers=args.workers)
        return dl

    train_dl, valid_dl, test_dl = map(_prepare_ds, ('train')), None, None
    # train_dl, valid_dl, test_dl = map(_prepare_ds, ('train', 'validation', 'test'))
    train_dl = _prepare_ds('train')

    return train_dl, valid_dl, test_dl


def get_correct_alignement(context, answer):
    """ Some original examples in SQuAD have indices wrong by 1 or 2 character. We test and fix this here. """
    gold_text = answer['text'][0]
    start_idx = answer['answer_start'][0]
    end_idx = start_idx + len(gold_text)
    if context[start_idx:end_idx] == gold_text:
        return start_idx, end_idx       # When the gold label position is good
    elif context[start_idx-1:end_idx-1] == gold_text:
        return start_idx-1, end_idx-1   # When the gold label is off by one character
    elif context[start_idx-2:end_idx-2] == gold_text:
        return start_idx-2, end_idx-2   # When the gold label is off by two character
    else:
        raise ValueError()


# Tokenize our training dataset
def convert_to_features(example_batch):
    # Tokenize contexts and questions (as pairs of inputs)
    input_pairs = list(zip(example_batch['context'], example_batch['question']))
    encodings = tokenizer.batch_encode_plus(input_pairs, pad_to_max_length=True, return_token_type_ids=True)

    # Compute start and end tokens for labels using Transformers's fast tokenizers alignement methodes.
    start_positions, end_positions = [], []
    for i, (context, answer) in enumerate(zip(example_batch['context'], example_batch['answers'])):
        start_idx, end_idx = get_correct_alignement(context, answer)
        start_positions.append(encodings.char_to_token(i, start_idx))
        end_positions.append(encodings.char_to_token(i, end_idx-1))
    encodings.update({'start_positions': start_positions,
                    'end_positions': end_positions})

    return encodings


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='QA Data')
    parser.add_argument('--qa_model', type=str, default='distilroberta-base', help='Model name')

    args = parser.parse_args()
    args.workers, args.percent, args.bs, args.fast_dev_run = 10, 100, 100, True
    prepare_data(args)
