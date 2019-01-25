import torch
from toolz import merge_with, first, merge


def stack_dicts(seq):
    return merge_with(lambda x: torch.stack(x, dim=1), seq)


def select_time(batch, i):
    out = {}
    for key, val in batch.items():
        if val.dim() == 1:
            out[key] = val
        else:
            out[key] = val[i, :]
    return out


def get_batch_size(batch):
    return first(batch.values()).size(0)


def concat_dicts(seq, dim=1):
    return merge_with(lambda x: torch.cat(x, dim=dim), *seq)


def batch_to_model_inputs(batch, aux, prog, diag, forcing, constants):
    """Prepare batch for input into model

    This function reshapes the inputs from (batch, time, feat) to (time, z, x,
    y) and includes the constants

    """

    batch = merge(batch, constants)

    def redim(val, num):
        if num == 1:
            return val.t().unsqueeze(-1)
        else:
            return val.permute(1, 2, 0).unsqueeze(-1)

    # entries in inputs and outputs need to be tuples
    data_fields = set(map(tuple, aux + prog + diag))

    out = {key: redim(batch[key], num) for key, num in data_fields}

    for key, num in prog:
        if key in forcing:
            out['F' + key] = redim(batch['F' + key], num)

    return out


def centered_difference(x, dim=-3):
    """Compute centered difference with first order difference at edges"""
    x = x.transpose(dim, 0)
    dx = torch.cat(
        [
            torch.unsqueeze(x[1] - x[0], 0), x[2:] - x[:-2],
            torch.unsqueeze(x[-1] - x[-2], 0)
        ],
        dim=0)
    return dx.transpose(dim, 0)


def get_other_dims(x, dim):
    return set(range(x.dim())) - {dim}


def mean_over_dims(x, dims):
    """Take a mean over a list of dimensions keeping the as singletons"""
    for dim in dims:
        x = x.mean(dim=dim, keepdim=True)
    return x


def mean_other_dims(x, dim):
    """Take a mean over all dimensions but the one specified"""
    other_dims = get_other_dims(x, dim)
    return mean_over_dims(x, other_dims)
