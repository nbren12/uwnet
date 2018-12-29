import pytest
import torch

from .groupby import DispatchByVariable, GroupBy


def test_dispatch_by_variables_membership():

    bins = [0, 1, 2, 3]
    expected = [0, 1, 3, 2, 4]
    a = torch.tensor(expected).float() - .5
    x = {'a': a.view(-1, 1)}
    model = DispatchByVariable(bins, bins, 'a', 0)
    membership = model.get_bin_membership(x)
    expected = [0, 1, 3, 2, 4]
    assert membership.tolist() == expected


@pytest.mark.parametrize('batch_shape', [(1, 3), (3, 1), (3, ), (1, 1, 3)])
def test_GroupBy_multiple_batch_dims(batch_shape):
    def fun(args):
        i, x = args
        return (i + 1) * x

    memberships = torch.tensor([0, 0, 1]).view(*batch_shape)
    x = torch.tensor([0, 1, 2]).view(*(*batch_shape, 1))
    out = GroupBy(memberships).apply(fun, x)

    assert out.shape == x.shape
    assert out.view(-1).tolist() == [0, 1, 4]


@pytest.mark.parametrize('memb', [
    [0, 1, 0],
    [1, 0, 0],
    [1, 2, 0],
])
def test_groupby_query(memb):
    memb = torch.tensor(memb)
    x = torch.tensor([
        0,
        1,
        2,
    ]).view(-1, 1)
    g = GroupBy(memb)
    y = torch.rand(3, 10, 3)
    out = g.apply(lambda x: y[x[0]].unsqueeze(0),
                  x)
    assert out.tolist() == y[memb].tolist()
