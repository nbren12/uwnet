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

from train_nn import train_model
from stewart_intro.evaluate_model import (
    plot_q_vs_nn_output,
    get_diagnostic_r2_score,
)
import numpy as np

min_time_step = .125
graph_save_location_format = '/Users/stewart/Desktop/time_error_dt_{}'


for dt in np.arange(min_time_step, 10 * min_time_step, min_time_step):
    print(f'\n\ndt = {dt}')
    model, data = train_model(dt=dt, model_name=f'dt_{dt}' + '_{}', n_epochs=1)
    plot_q_vs_nn_output(
        model,
        data,
        save_location_format_str=graph_save_location_format.format(
            dt) + '{}.png')
    get_diagnostic_r2_score(model, data, dt=dt)
