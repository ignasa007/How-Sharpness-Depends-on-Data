import torch

from utilities import MLP


DATASTORE = "./datasets"
RESULTS = "./results"
ASSETS = "./assets"

SUBSET_SIZE = 5000
SIGMA_X = 1.

LOSS_TYPE = "Mean Squared Error"
EPOCHS = 10000
LOG_EVERY = 500

MODEL = MLP
WIDE_SHALLOW_WIDTHS = [256]*1
NARROW_DEEP_WIDTHS = [64]*3
MODEL_KWARGS = dict()

ACTIVATION = "Linear"
ACT_KWARGS = dict()

WIDE_SHALLOW_LR = 1e-3
NARROW_DEEP_LR = 3e-3
SGD_KWARGS = dict()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0

HUTCHINSON_SAMPLES = 100

AX_WIDTH = 6.4
AX_HEIGHT = 5.4
LABEL_SIZE = 18
FONT_SIZE = 21