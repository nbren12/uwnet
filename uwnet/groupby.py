import torch
from torch import nn
from toolz import first, pipe


def bucketize(tensor, bucket_boundaries):
    """Equivalent to numpy.digitize

    Notes
    -----
    Torch does not have a built in equivalent yet. I found this snippet here:
    https://github.com/pytorch/pytorch/issues/7284
    """
    result = torch.zeros_like(tensor, dtype=torch.int32)
    for boundary in bucket_boundaries:
        result += (tensor > boundary).int()
    return result


def _unflatten(output, shape):
    output_shape = shape + output.shape[1:]
    return output.view(*output_shape)


class GroupBy(object):
    def __init__(self, memberships):
        self.shape = memberships.shape
        self.memberships = memberships.view(-1)
        self.compute_group_indices()

    @property
    def n(self):
        return self.memberships.numel()

    @property
    def groups(self):
        return self.group_indices.keys()

    def compute_group_indices(self):
        num_indices = 0
        self.group_indices = {}
        k = 0
        while num_indices < self.n:
            inds = torch.nonzero(self.memberships == k).view(-1)
            self.group_indices[k] = inds
            num_indices += inds.shape[0]
            k += 1

    def _apply_over_groups(self, fun, x):
        outputs = {}
        for k, inds in self.group_indices.items():
            args = (k, x.index_select(0, inds))
            outputs[k] = fun(args)
        return outputs

    def _init_output_array(self, outputs):
        nout = first(outputs.values()).shape[1:]
        dtype = first(outputs.values()).dtype
        return torch.zeros(self.n, *nout, dtype=dtype)

    def _gather_from_group_outputs(self, outputs):
        output = self._init_output_array(outputs)
        for k in outputs:
            output[self.group_indices[k]] = outputs[k]
        return output

    def _unflatten(self, output):
        return _unflatten(output, self.shape)

    def flatten_input(self, x):
        m = len(self.shape)
        if x.shape[:m] != self.shape:
            raise ValueError(
                "memberships and x must match along the first few "
                "dimensions")

        if x.dim() <= m:
            raise ValueError("x must have more dimensions than memberships")

        shape = x.shape
        x = x.view(self.n, *shape[m:])
        return x

    def apply(self, fun, x):
        """Apply function over groups and combine output

        Parameters
        ----------
        fun : callable
            Takes (i, x) as an argument, output shape must include the group
            dimension.
        x : torch.tensor
            Argument to pass to fun. The first dimensions must match the
            membership array.

        Returns
        -------
        merged_output : torch.tensor
        """
        x = self.flatten_input(x)
        group_output = self._apply_over_groups(fun, x)
        return pipe(group_output, self._gather_from_group_outputs,
                    self._unflatten)


class DispatchByVariable(nn.Module):
    """Dispatch

    """

    def __init__(self, bins, objs, variable, index):
        super(DispatchByVariable, self).__init__()
        self.bins = bins
        self.objs = objs
        self.variable = variable
        self.index = index

    def get_binning_variable(self, x):
        return x[self.variable][..., self.index]

    def get_bin_membership(self, x):
        y = self.get_binning_variable(x)
        return bucketize(y, self.bins)

    def forward(self, x):
        memberships = self.get_bin_membership(x)
        return GroupBy(memberships, x)
