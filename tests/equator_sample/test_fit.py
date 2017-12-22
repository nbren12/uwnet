from lib.models.torch.preprocess import prepare_data
from lib.models.torch.multiple_step_objective import train_multistep_objective
import torch



data = prepare_data("data/prog/*.nc", "data/forcing/*.nc", "data/w.nc",
                    subset_fn=lambda x: x)
net = train_multistep_objective(data, num_epochs=2, weight_decay=.1,
                                window_size=10,
                                learning_rate=.005)

torch.save(net, "net.torch")
