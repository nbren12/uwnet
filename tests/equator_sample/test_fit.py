from lib.preprocess import ForcedData
from lib.models.multiple_step_objective import train_multistep_objective
import torch



data = ForcedData.from_files("data/prog/*.nc", "data/forcing/*.nc",
                             "data/w.nc")

net = train_multistep_objective(data, num_epochs=2, weight_decay=.1,
                                learning_rate=.005)

torch.save(net, "net.torch")
