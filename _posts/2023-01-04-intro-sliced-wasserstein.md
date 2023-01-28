---
title: Introduction to Sliced Wasserstein
date: 2023-01-04
categories: [AI, Sliced Wasserstein]
# layout: posts
tags: [AI, Sliced Wasserstein, divergence]
---
One of the main goals of generative AI is to solve a model that follows the data's distribution. A common approach is to compare probability distributions and iteratively minimize the distance between the two. To compare probability distributions, we can employ the concept of [divergence](https://en.wikipedia.org/wiki/Divergence_(statistics)). KL-Divergence, one of many divergence measures, is routinely cited in literature and has applications in variational inference and GANs. KL-divergence has its drawbacks, however. For instance, the distance between two distributions on non-overlapping domains is infinite and may be computationally challenging to approximate even when overlapping [Nadjahi][1]. 

An alternative function to evaluate is the [Wasserstein Distance](https://en.wikipedia.org/wiki/Wasserstein_metric) (WD). Formally, it's defined as 

$$ W_p(\mu, \nu) = (\underset{\pi\in\Pi(\mu, \nu)}{\inf} \int_{\textsf{X}\times\textsf{X}} \rho(x, y)^p d\pi(x, y) )^{1/p} $$

where $$\rho$$ is a valid metric, $$\pi$$ is the joint probability distribution between $$x, y$$ and generally $$p=2$$. We can interpret this as the optimal cost of transporting the probability density from one measure to another. While the formula looks intractable, let's consider univariate distributions. The above equation reduces to 

$$ W_p^p(\mu, \nu) =\int_{0}^{1}| F_\mu^{-1}(t) - F_\nu^{-1}(t) |^p dt  $$

where $$F$$ is a quantile function. We can push further, and approximate $$F$$ with a Monte Carlo approach

$$ W_p^p(\mu, \nu) \approx \frac{1}{K} \sum_{k=1}^K | {\overset{\sim}{F}}_\mu^{-1}(t_k) - \overset{\sim}{F}_\nu^{-1}(t_k) |^p$$ 

and in the case of univariate distributions, the quantile function can be replaced with empirical sample sorting. However, in large-scale AI the Wasserstein Distance is generally analytically intractable, and prohibitively expensive, $$O(n^3 log(n))$$ [Nadjahi][1].


Enter Sliced Wasserstein Distance (SWD).

$$ \boldsymbol{SW}_p^p = \int_{S^{d-1}} \boldsymbol{W}_p^p(u_{\#}\mu, u_{\#}\nu) d\sigma(u)$$

where $$u_\#$$ denotes the push-forward operator associated with $$u$$, $$\sigma$$ denotes the uniform distribution on $$S^{d-1}$$, and $$S^{d-1}$$ is the set of unit hyperspheres in $$\mathbb{R}^d$$ where $$u \in S^{d-1}$$ [Special Orthogonal Group](https://en.wikipedia.org/wiki/Orthogonal_group#Special_orthogonal_group). The push-forward operator can take many forms but the linear form is defined as $$u_\# \mu = \langle u, \mu \rangle $$ which is the inner product. Alternatively stated, the SWD is the expectation of the WD of two distributions (pushed forward by an element on the unit hypersphere) with respect to the uniform distribution on the hypersphere. The integral of the unit hypersphere is generally not feasible to calculate, but the SWD can also be approximated with a Monte Carlo approach by picking $$L$$ elements on the unit hypersphere and averaging

$$ \boldsymbol{SW}_p^p = \frac{1}{L} \sum_{l=1}^{L} \boldsymbol{W}_p^p(u_{l\#}\mu, u_{l\#}\nu) $$ 

Let's look at a toy example to illustrate how it works concretely.

#### Initial Setup
This section of code show the imports, function defintions, and initial setup of the training routine.
```python
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll

def uniform_hypersphere(D, L, device='cpu', dtype=torch.float):
    """
        return L, D-dimensional hyperspheres
    """
    
    hypersphere = torch.randn(D, L, device=device, dtype=dtype)
    hypersphere = hypersphere / torch.norm(hypersphere, dim=0, keepdim=True)
    return hypersphere

def push_forward(u, x):
    """
        linear push-forward measure
        push forward x by u, where u comes from uniform_hypersphere
        
    """
    return x@u

def sliced_wasserstein_distance(x, y, L, loss='l1'):
    """
        Compute sw distance
    """
    device = y.device
    D = y.shape[-1]
    directions = uniform_hypersphere(D, L, device)
    x_push_forward = push_forward(directions, x)
    y_push_forward = push_forward(directions, y)
    
    x_sort = torch.sort(x_push_forward, dim=-2)
    y_sort = torch.sort(y_push_forward, dim=-2)
    
    if loss == 'l1':
        return torch.nn.functional.l1_loss(x_sort.values, y_sort.values)
    elif loss == 'mse':
        return torch.nn.functional.mse_loss(x_sort.values, y_sort.values)
    else:
        raise NotImplementedError

def plot_points(set1, set2, title):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        set1[:, 0], set1[:, 1], set1[:, 2], c='red', s=50, alpha=0.5
    )
    ax.scatter(
        set2[:, 0], set2[:, 1], set2[:, 2], c=sr_color, s=50, alpha=0.5
    )

    ax.set_title(title)

    ax.view_init(azim=70, elev=5)
    plt.show()


DEVICE = 'cuda:1'
n_points = 1500
iterations = 200
sr_points, sr_color = make_swiss_roll(
    n_samples=n_points, random_state=0
)

swiss_roll_points = torch.tensor(sr_points).to(torch.float)
parameters = torch.rand_like(swiss_roll_points).requires_grad_(True)

plot_points(
    parameters.detach().numpy(),
    sr_points,
    "Swiss Roll in Ambient Space + Init Distribution"
)
```
{% include figure image_path="/assets/images/intro-sliced-wasserstein_files/intro-sliced-wasserstein_2_0.png" alt="Graph Showing initial distribution of parameters and swiss roll" caption="Initial Distribution" %}


#### Move Points
This last section shows how to run the actual optimization.
```python
optimizer = torch.optim.Adam([parameters], lr=1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, iterations, eta_min=1e-1
)
for i in range(iterations):
    optimizer.zero_grad()
    l = sliced_wasserstein_distance(
        parameters, swiss_roll_points, 50, 'l1'
    )
    l.backward()
    optimizer.step()
    scheduler.step()
    
plot_points(
    parameters.detach().numpy(),
    sr_points,
    "Swiss Roll in Ambient Space + Final Distribution"
)
```

{% include figure image_path="/assets/images/intro-sliced-wasserstein_files/intro-sliced-wasserstein_3_0.png" alt="Graph Showing the final distribution of parameters and swiss roll" caption="Final Distribution" %}


The above code shows how to sample uniform directions from the unit hypersphere, project data points onto it, and calculate the SWD. There is, however, a drawback. We still need to decide on the number of directions to use. Increase the number for a more accurate distance metric. But, some projections map points that should be far away, close to each other. We must choose enough directions so that we get a robust signal to backpropagate. In high-dimensional problems, this becomes a potential bottleneck. In my next post, I'll discuss a way to alleviate this issue.



[1]: https://hal.science/tel-03533097

