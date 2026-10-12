from typing import Optional, Callable

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
import torch
from torch import autograd, nn, optim
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector
from torchvision import transforms
from torch.utils.data import Subset

from constants import DATASTORE


def get_activation(act_name: str, **act_kwargs) -> nn.Module:

    if act_name is None:
        return nn.Identity()

    act_name = act_name.lower()
    if act_name == "identity" or act_name == "linear":
        return nn.Identity()
    if act_name == "relu":
        return nn.ReLU()
    if act_name == "leakyrelu" or act_name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=act_kwargs.get("negative_slope", 0.01))
    if act_name == "tanh":
        return nn.Tanh()
    if act_name == "sigmoid":
        return nn.Sigmoid()

    raise ValueError(f"Don't recognize `act_name={act_name}`.")

class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def _all_layers(self) -> list[nn.Module]:
        return self.emb_layers + [self.readout]

    def reset_parameters(self, seed: int = None) -> None:
        torch.manual_seed(seed)
        for layer in self._all_layers():
            layer.reset_parameters()

    def prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class MLP(Model):

    def __init__(
        self, widths: list[int], act_name: str, act_kwargs: dict, output_dim: int,
        normalization: bool = False,
    ):
        super().__init__()
        self.emb_layers = nn.ModuleList([nn.LazyLinear(width, bias=False) for width in widths])
        self.activation = get_activation(act_name, **act_kwargs)
        self.normalization = nn.LayerNorm(widths[-1], eps=1e-12, elementwise_affine=False) if normalization else nn.Identity()
        self.readout = nn.LazyLinear(output_dim, bias=False)

    @torch.no_grad()
    def prepare_input(self, x: torch.Tensor, sigma_x: Optional[float] = None) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        self.forward(x[:1])     # Initialize lazy layers
        if sigma_x is not None:
            assert x.size(0) >= x.size(1)
            u, s, vt = torch.linalg.svd(x, full_matrices=False)
            x = (u * sigma_x*np.sqrt(u.size(0))) @ vt
        return x

    def balanced_init(self, seed: int = None) -> None:
        # Set seed
        torch.manual_seed(seed)
        # Setup
        all_layers = self._all_layers()
        d = all_layers[0].in_features
        k = min([d] + [l.out_features for l in all_layers])
        min_norm_sq = min([l.weight.data.norm().square().item() for l in all_layers])
        # Sample singular values
        s_sq = torch.sort(torch.rand(k), descending=True).values
        s_sq = s_sq * min_norm_sq/s_sq.sum()        # Set the norm of each layer equal to the minimum norm
        S = torch.diag(torch.sqrt(s_sq))
        # Sample weight matrices
        V, _ = torch.linalg.qr(torch.randn(d, k))
        for layer in all_layers:
            U, _ = torch.linalg.qr(torch.randn(layer.out_features, k))
            with torch.no_grad():
                W = U @ S @ V.T
                layer.weight.copy_(W)
            V = U

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for emb_layer in self.emb_layers:
            x = self.activation(emb_layer(x))
        x = self.normalization(x)
        return self.readout(x)

    def compute_balance(self, balance_op: Callable = torch.linalg.norm) -> list:
        all_layers = self._all_layers()
        out = list()
        for idx in range(len(all_layers)-1):
            out.append(balance_op(
                all_layers[idx].weight.data @ all_layers[idx].weight.data.T
                - all_layers[idx+1].weight.data.T @ all_layers[idx+1].weight.data
            ))
        return out

def loss_fn(outputs, one_hot, loss_type):
    if loss_type.lower() == "cross entropy loss":
        return F.cross_entropy(outputs, one_hot, reduction="sum") / outputs.size(0)
    elif loss_type.lower() == "mean squared error":
        return 0.5 * F.mse_loss(outputs, one_hot, reduction="sum") / outputs.size(0)

def loss_gradient(outputs, one_hot, loss_type):
    if loss_type.lower() == "cross entropy loss":
        return (torch.softmax(outputs, dim=1) - one_hot) / outputs.size(0)
    elif loss_type.lower() == "mean squared error":
        return (outputs - one_hot) / outputs.size(0)

def alignment(outputs, one_hot, loss_type):
    return (outputs * -loss_gradient(outputs, one_hot, loss_type)).sum().item()

def sq_parameter_norm(model):
    return [p.norm().square().item() for p in model.parameters()]

def compute_gradient(X, Y, model):
    model.zero_grad()
    loss = loss_fn(model(X), Y, loss_type="Mean Squared Error")
    gradient = parameters_to_vector(autograd.grad(loss, inputs=model.parameters()))
    return gradient

def compute_hvp(X, Y, model, vector, device):
    model.zero_grad()
    vector = vector.to(device)
    loss = loss_fn(model(X), Y, loss_type="Mean Squared Error")
    grad = autograd.grad(loss, model.parameters(), create_graph=True)
    dot = parameters_to_vector(grad).mul(vector).sum()
    hvp = autograd.grad(dot, model.parameters())
    hvp = parameters_to_vector([v.contiguous() for v in hvp])
    return hvp

def lanczos(tensor_matvec, p, n_eigs):
    def array_matvec(vec: np.ndarray):
        return tensor_matvec(torch.tensor(vec))
    operator = LinearOperator((p, p), matvec=array_matvec)
    eig_vals, eig_vecs = eigsh(operator, k=n_eigs, which="LA")
    return torch.from_numpy(np.ascontiguousarray(eig_vals[::-1]).copy()), \
        torch.from_numpy(np.ascontiguousarray(np.flip(eig_vecs, -1)).copy())

def train_gd(
    Dataset,
    output_dim: int,
    subset_size: int,
    sigma_x: Optional[float],
    max_prop: Optional[float],
    Model: nn.Module,
    widths: list[int],
    act_name: str,
    act_kwargs: dict,
    model_kwargs: dict,
    balanced_init: bool,
    loss_type: str,
    lr: float,
    optim_kwargs: dict,
    epochs: int,
    log_every: int,
    device: torch.device,
    seed: int,
    balance_op: Optional[Callable] = None,
    tracker_fns: list[Callable] = [],
):

    full_trainset = Dataset(
        root=f"{DATASTORE}/{Dataset.__name__.lower()}",
        train=True, download=True, transform=transforms.ToTensor()
    )
    if max_prop is not None:
        full_trainset.targets = torch.tensor(full_trainset.targets)
        sizes = [int(subset_size*(1-max_prop)/(output_dim-1))]*(output_dim-1)
        sizes = [subset_size-sum(sizes)] + sizes
        indices = list()
        for label, size in zip(range(output_dim), sizes):
            indices += torch.where(full_trainset.targets==label)[0][:size].tolist()
        full_trainset.targets = full_trainset.targets.tolist()
    else:
        indices = list(range(subset_size))
    train_subset = Subset(full_trainset, indices)

    data_loader = torch.utils.data.DataLoader(train_subset, batch_size=subset_size)
    images, labels = next(iter(data_loader))
    images, labels = images.to(device), labels.to(device)
    one_hot = torch.nn.functional.one_hot(labels, num_classes=output_dim).float()

    model = Model(
        widths=widths, output_dim=output_dim,
        act_name=act_name, act_kwargs=act_kwargs,
        **model_kwargs,
    ).to(device)

    train_inputs = model.prepare_input(images, sigma_x=sigma_x)      # Also initializes lazy layers
    model.reset_parameters(seed=seed)
    if balanced_init:
        model.balanced_init(seed=seed)

    optimizer = optim.SGD(model.parameters(), lr=lr, **optim_kwargs)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=100, threshold=1e-3, threshold_mode="rel"
    )

    losses, param_norms = [], []
    balances = [model.compute_balance(balance_op)] if balance_op is not None else None
    tracked_vals = [[]]
    for tracker_fn in tracker_fns:
        tracked_vals[-1].append(tracker_fn(train_inputs, one_hot, model, device))

    for epoch in range(1, epochs+1):

        model.train()
        optimizer.zero_grad()
        outputs = model(train_inputs)
        loss = loss_fn(outputs, one_hot, loss_type)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.detach().item())

        losses.append(loss.detach().item())
        param_norms.append(sq_parameter_norm(model))
        if balance_op is not None:
            balances.append(model.compute_balance(balance_op))

        if epoch % log_every == 0:
            _, predicted = torch.max(outputs.data, 1)
            acc = (predicted == labels).float().mean().item() * 100
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}, Acc: {acc:.2f}%')
            tracked_vals.append([])
            for tracker_fn in tracker_fns:
                tracked_vals[-1].append(tracker_fn(train_inputs, one_hot, model, device))

    assets = (train_inputs, one_hot, model)
    out = list(map(np.array, (losses, param_norms)))
    out.append(np.array(balances) if balance_op is not None else None)
    out.extend(list(map(np.array, zip(*tracked_vals))))

    return assets, tuple(out)