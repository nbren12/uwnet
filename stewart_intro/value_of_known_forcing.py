from train_nn import train_model
from stewart_intro.evaluate_model import (
    plot_q_vs_nn_output,
    get_diagnostic_r2_score,
)

graph_save_location_format = '/Users/stewart/Desktop/known_forcing_{}.png'


for include_known_forcing in [False, True]:
    model, data = train_model(
        model_name=f'include_known_forcing_{include_known_forcing}',
        include_known_forcing=include_known_forcing,
        n_epochs=1)
    plot_q_vs_nn_output(
        model, data, save_location=graph_save_location_format.format(
            include_known_forcing) + '_{}',
        include_known_forcing=include_known_forcing)
    get_diagnostic_r2_score(
        model, data, include_known_forcing=include_known_forcing)
