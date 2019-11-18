from uwnet.wave import LinearResponseFunction, base_from_xarray, model_plus_damping
import xarray as xr
import torch
from typing import Mapping
from uwnet.tensordict import TensorDict
from uwnet.jacobian import dict_jacobian

def torch2numpy(x):
    if isinstance(x, Mapping):
        return {key: torch2numpy(val) for key, val in x.items()}
    else:
        return x.detach().numpy()


def numpy2torch(x):
    if isinstance(x, Mapping):
        return {key: numpy2torch(val) for key, val in x.items()}
    else:
        return torch.tensor(x, requires_grad=True).float()


def get_test_solution(base_state):
    from collections import namedtuple

    s, q = base_state["SLI"].detach(), base_state["QT"].detach()
    s.requires_grad = True
    q.requires_grad = True
    w = torch.zeros(s.size(0), requires_grad=True)
    soln = namedtuple("Solution", ["w", "s", "q"])
    return soln(w=w, s=s, q=q)


def from_model(src, base_state):
    """Compute jacobian matrix about a given numpy base state


    Parameters
    ----------
    src
        torch module of the parametrized sources. predicts outputs in
        "per day" units.
    base_state : dict
        dict of numpy arrays providing the base state

    Returns
    -------
    lrf : dict
        dict of numpy arrays giving the linearized reponse function
    """
    base_state = numpy2torch(base_state)
    sol = get_test_solution(base_state)
    outs = {}
    if src is not None:
        srcs = src(
            TensorDict(
                {
                    "SLI": sol.s,
                    "QT": sol.q,
                    "SST": base_state["SST"],
                    "SOLIN": base_state["SOLIN"],
                }
            )
        )
        outs["s"] = srcs["SLI"] / 86400
        outs["q"] = srcs["QT"] / 86400

    ins = sol._asdict()
    jac = dict_jacobian(outs, ins)
    numpy_jac = torch2numpy(jac)
    return LinearResponseFunction(numpy_jac)

path = "../../nn/NNLowerDecayLR/20.pkl"
src = torch.load(path)
mean = xr.open_dataset("../../data/processed/training.mean.nc")
eq_mean = mean.isel(y=32)

src = model_plus_damping(src)
base_state = base_from_xarray(eq_mean)
lrf = from_model(src, base_state)

with open("lrf.json", "w") as f:
    lrf.dump(f)
