import matplotlib.pyplot as plt
from train_nn import train_model

min_time_step = .125
graph_save_location_format = '/Users/stewart/Desktop/time_error_dt_{}'

errors = []
hidden_nodes = []
sizes = [1000, 2500, 10000, 25000, 100000]
for n_hidden_nodes in sizes:
    model, data, normalization_dict, error = train_model(
        model_name=f'n_hidden_nodes_{n_hidden_nodes}' + '_{}',
        n_epochs=5,
        n_hidden_nodes=n_hidden_nodes)
    errors.append(error)
    hidden_nodes.append(n_hidden_nodes)

plt.scatter(sizes, errors)
plt.xlabel('N Hidden Nodes')
plt.ylabel('MSE')
plt.title('Model Complexity Error')
plt.show()
plt.savefig('/Users/stewart/Desktop/n_hidden_nodes_vs_error.png')
