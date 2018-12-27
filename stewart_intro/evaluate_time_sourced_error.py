"""
We wish to address the question:
    What portion of the inaccuracy of the model is due to the large time step?

Perhaps the reason we can't get much better than an R2 of 0.5 is because a 3
hour time step is too large for the neural network to predict the correction to
the SAM model. If it was something like 5 minutes, perhaps the R2 would be much
larger.

We can evaluate this hypothesis by training models with 3, 6, and 9 hour time
steps, and evaluating the R2 for each step.
"""
import matplotlib.pyplot as plt
from train_nn import train_model
from stewart_intro.evaluate_model import (
    plot_model_error_output,
    get_diagnostic_r2_score,
)
import numpy as np

min_time_step = .125
graph_save_location_format = '/Users/stewart/Desktop/time_error_dt_{}'

errors = []
dts = []
for dt in np.arange(min_time_step, 10 * min_time_step, min_time_step):
    print(f'\n\ndt = {dt}')
    model, data, normalization_dict, error = train_model(
        dt=dt, model_name=f'dt_{dt}' + '_{}', n_epochs=2)
    errors.append(error)
    dts.append(dt)
    plot_model_error_output(
        model,
        data,
        normalization_dict,
        save_location_format_str=graph_save_location_format.format(
            dt) + '{}.png')
    get_diagnostic_r2_score(model, data, normalization_dict, dt=dt)

plt.scatter(dts, errors)
plt.xlabel('dt (days)')
plt.ylabel('MSE')
plt.title('Time Sourced Error')
plt.show()
plt.savefig('/Users/stewart/Desktop/time_sourced_error.png')
