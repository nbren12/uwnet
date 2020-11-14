import torch
from torch.autograd import grad
from .xarray_interface import dataset_to_torch_dict
import numpy as np


def jacobian_from_xarray(model, ds):
    torch_data = dataset_to_torch_dict(ds)
    torch_data = torch_data[model.input_names]

    # compute saliency map
    jac = jacobian_from_model(model, torch_data)

    return jac


def dict_format_to_numpy(jac):
    out = []
    for outkey in jac:
        row = []
        for inkey in jac:
            arr = jac[outkey][inkey].detach().numpy()
            row.append(arr)
        out.append(row)
    return np.block(out)


def jacobian_backward(y, x):
    """Back-propagates the Frobenious norm of the jacobian"""
    n = len(y)
    out = 0.0
    for i in range(n):
        y_x = grad(y[i], x, create_graph=True)[0]
        y_x2 = y_x.norm() ** 2 / 2
        y_x2.backward(retain_graph=True)
        out += y_x2.item()
    return out


def jacobian_norm(y, x):
    n = len(y)
    out = 0.0
    for i in range(n):
        y_x = grad(y[i], x, create_graph=True)[0]
        out += y_x.norm() ** 2 / 2
    return out


def jacobian(y, x):
    """Jacobian of y with respect to x

    Returns
    -------
    jacobian : torch.tensor
        jacobian[i,j] is the partial derivative of y[i] with respect to x[j].
    """
    n = len(y)
    jac = []
    for i in range(n):
        y_x = grad(y[i], x, create_graph=True, allow_unused=True)[0]
        if y_x is None:
            y_x = torch.zeros(x.size(0))
        jac.append(y_x)
    return torch.stack(jac)


def max_eig_val(A, niter=10, m=1):
    """

    Parameters
    ----------
    A : matrix
    niter : number of iterations of power method
    m :
        number of iterations to keep gradients from end to keep gradients for
    """
    n = A.size(0)
    x = torch.rand(n)
    for i in range(niter):
        if i < niter - m:
            x = x.detach()
        y = A.matmul(x)
        norm = x.norm()
        lam = y.dot(x) / norm / norm
        x = y / lam / norm
    return lam, x


def max_signed_eigvals(A, niter=100, m=1):

    # find maximum norm eigvalue
    lam, _ = max_eig_val(A, niter=niter, m=m)
    # if it is negative shift the matrix
    h = -1 / lam * 0.9

    I = torch.eye(A.size(0))
    B = I + h * A

    lam_plus, _ = max_eig_val(B, niter=niter, m=m)
    lam_orig = (lam_plus - 1) / h

    if lam.item() < lam_orig.item():
        lam, lam_orig = lam_orig, lam
    return lam, lam_orig


def dict_jacobian(y, d, progs=["QT", "SLI"]):
    """Compute Jacobian dictionary

    Returns
    -------
    jacobian : dict of dicts
        jacobian[outkey][inkey][i, j] is the sensitivity of
        outkey[i] with respect to inkey[j]
    """
    jac = {}
    for inkey in d:
        for outkey in y:
            try:
                jac.setdefault(outkey, {})[inkey] = jacobian(
                    y[outkey], d[inkey]
                ).squeeze()
            except KeyError:
                pass
    return jac


def jacobian_from_model(model, d, **kwargs):
    # enable gradients
    for key in d:
        d[key].requires_grad = True
    y = model(d)
    return dict_jacobian(y, d, **kwargs)
