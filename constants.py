import torch


DATASTORE = "./datasets"
ASSETS = "./assets"
SUBSET_SIZE = 5000
EPOCHS = 10000; LOG_EVERY = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0
HUTCHINSON_SAMPLES = 100