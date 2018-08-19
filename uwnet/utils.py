import torch
from toolz import merge_with, first


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
    return first(batch.values()).size(0)


def concat_dicts(seq, dim=1):
    return merge_with(lambda x: torch.cat(x, dim=dim), *seq)
