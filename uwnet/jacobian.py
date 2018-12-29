import torch
from torch.autograd import grad


def jacobian_backward(y, x):
    """Back-propagates the Frobenious norm of the jacobian"""
    n = len(y)
    out = 0.0
    for i in range(n):
        y_x = grad(y[i], x, create_graph=True)[0]
        y_x2 = y_x.norm()**2 / 2
        y_x2.backward(retain_graph=True)
        out += y_x2.item()
    return out


def jacobian_norm(y, x):
    n = len(y)
    out = 0.0
    for i in range(n):
        y_x = grad(y[i], x, create_graph=True)[0]
        out += y_x.norm()**2 / 2
    return out


def jacobian(y, x):
    n = len(y)
    jac = []
    for i in range(n):
        y_x = grad(y[i], x, create_graph=True)[0]
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
    h = - 1 / lam * .9

    I = torch.eye(A.size(0))
    B = I + h * A

    lam_plus, _ = max_eig_val(B, niter=niter, m=m)
    lam_orig = (lam_plus - 1) / h

    if lam.item() < lam_orig.item():
        lam, lam_orig = lam_orig, lam
    return lam, lam_orig


def dict_jacobian(y, d, progs=['QT', 'SLI']):
    for key in d:
        try:
            d[key].requires_grad = True
        except RuntimeError:
            pass

    jac = {}
    for inkey in progs:
        for outkey in progs:
            try:
                jac.setdefault(inkey, {})[outkey] = jacobian(
                    y[inkey], d[outkey]).squeeze()
            except KeyError:
                pass
    return jac


def jacobian_from_model(model, d, **kwargs):
    y = model(d)
    return dict_jacobian(y, d, **kwargs)
