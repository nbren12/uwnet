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

from stewart_intro.simple_nn import train_model
from stewart_intro.examine_simple_nn import (
    plot_q_vs_nn_output,
    get_diagnostic_r2_score,
)
import numpy as np

min_time_step = .125
graph_save_location_format = '/Users/stewart/Desktop/time_error_dt_{}.png'


for dt in np.arange(min_time_step, 4 * min_time_step, min_time_step):
    print(f'\n\ndt = {dt}')
    w1, w2, data = train_model(
        dt=dt, model_name=f'dt_{dt}' + '_{}', n_epochs=1)
    plot_q_vs_nn_output(
        w1, w2, data, save_location=graph_save_location_format.format(dt))
    get_diagnostic_r2_score(w1, w2, data, dt=dt)
