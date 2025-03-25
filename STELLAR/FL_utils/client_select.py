import numpy as np

def client_selection(num_clients, client_selection_ratio, client_selection_strategy, *args, **kwargs):
    np.random.seed(42)
    if client_selection_strategy == "random":
        num_selected = max(int(client_selection_ratio * num_clients), 1)
        selected_clients_set = set(np.random.choice(np.arange(num_clients), num_selected, replace=False))

    return selected_clients_set