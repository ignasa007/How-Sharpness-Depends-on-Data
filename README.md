# How the Hessian-Spectrum of Neural Networks Depends on Data

This repository contains the code to reproduce the results in our paper *How the Hessian-Spectrum of Neural Networks Depends on Data*, presented at the *Workshop on High Dimensional Learning Dynamics* at the *Internation Conference on Machine Learning*, 2026.

## Reproducing the Figures

```bash
# Figure 1
python eigvals_w_shallow_networks.py
# Figure 2
python comparing_bounds.py
# Figure 3
python sharpness_paramnorm.py
# Figure 4
python eigvals_w_deep_networks.py
# Figure 5
python effect_of_isotropy.py
# Figure 6
python balanced_network_random_data.py
python effect_of_depth.py
python sharpness_w_balanced_init.py
# Figure 7
python effect_of_dataset_size.py
# Figure 8
python shedding_assumption1.py
python shedding_assumption2.py
python effect_of_activation.py
```

## Anistropic Feature Covariance

We observe that for $\ell \geq 2$,
```math
    \lambda_{\max}\left(\frac{1}{n} X^T X\right) \lambda_{\max}\left((W_L W_L^T)^{L-1}\right)
    \gg \lambda_{\max}\left(\frac{1}{n} X^T (W_1^T W_1)^{\ell-1} X\right) \lambda_{\max}\left((W_L W_L^T)^{L-\ell}\right)
```
because the sharpness of the first factor in the right-hand expression vanishes, so that
```math
    \lambda_{\max}(NTK) \approx \frac{1}{n} \|X\|_2^2 \|W\|_2^{2(L-1)}
    = \frac{1}{n} \|X\|_2^2 \|YX^{\dagger}\|_2^{2(L-1)}
```

1\. *Why does the first term in the summation dominate? How does $W_1$ kill-off the sharpness brought in by $X$?*

2\. *How do I interpret $\|YX^{\dagger}\|_2$ to pinpoint what makes solutions to some datasets sharper than to others?*

3\. *Do we observe this at sub-optimal parameters, i.e. not OLS solutions?* Yes, this can be observed at random weights. I assume you need some constraint on them, but simply sampling $\mathcal{N}(0,1)$ entries is good enough.

## Effect of Anisotropy

1\. *The OLS loss does not depend on the singular value distribution. Is this obvious?* Yes, the OLS loss is given by
```math
L(W^*; X, Y) = \frac{1}{2n} \left\| (I_n-UU^T)Y \right\|_F^2
```
where $U$ collects the left singular vectors of $X$. Hence, it is independent of the singular values.

2\. *Why do the upper and lower bounds have an exponential trend? Part of the effect must be coming from the features, but is any of it also coming from the weights?* Yes, both increase with sampling variance. It is clear why $\|X\|_2$ increases, but it is hard to predict how $\|W\|_2$ changes since the singular values are not the same, i.e. Assumption 1 does not hold.

## Citation

```
@inproceedings{
    singh2026how,
    title={How the Hessian-Spectrum of Neural Networks Depends on Data},
    author={Jasraj Singh and Enea Monzio Compagnoni and Antonio Orvieto},
    booktitle={High-dimensional Learning Dynamics 2026},
    year={2026}
}
```