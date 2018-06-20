import torch
from toolz import merge_with


def stack_dicts(seq):
    return merge_with(lambda x: torch.stack(x, dim=1), seq)


def select_time(batch, i):
    out = {}
    for key, val in batch.items():
        if val.dim() == 1:
            out[key] = val
        else:
            out[key] = val[:, i]
    return out


def get_batch_size(batch):
    return batch['sl'].size(0)


def concat_dicts(seq):
    return merge_with(lambda x: torch.cat(seq, dim=1), *seq)
