"""$project_name: A Flower / NumPy app."""

import flwr as fl

# Configure the strategy
strategy = fl.server.strategy.FedAvg()

# Flower ServerApp
app = fl.server.ServerApp(
    config=fl.server.ServerConfig(num_rounds=1),
    strategy=strategy,
)
