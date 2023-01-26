#### Initial Setup
```python
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll

def uniform_hypersphere(D, L, device='cpu', dtype=torch.float):
    """
        return L, D-dimensional hyperspheres
    """
    
    hypersphere = torch.randn(D, L, device=device, dtype=dtype)
    hypersphere = hypersphere / torch.norm(hypersphere, p=2, dim=0, keepdim=True)
    return hypersphere

def pushforward(u, x):
    """
        linear pushforward measure
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
    x_pushforward = pushforward(directions, x)
    y_pushforward = pushforward(directions, y)
    
    x_sort = torch.sort(x_pushforward, dim=-2)
    y_sort = torch.sort(y_pushforward, dim=-2)
    
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

plot_points(parameters.detach().numpy(), sr_points, "Swiss Roll in Ambient Space + Init Distribution")
```


    
![png](/assets/images/intro-sliced-wasserstein_files/intro-sliced-wasserstein_2_0.png)
    


#### Move Points
```python
optimizer = torch.optim.Adam([parameters], lr=1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iterations, eta_min=1e-1)
for i in range(iterations):
    optimizer.zero_grad()
    l = sliced_wasserstein_distance(parameters, swiss_roll_points, 50, 'l1')
    l.backward()
    optimizer.step()
    scheduler.step()
    
plot_points(parameters.detach().numpy(), sr_points, "Swiss Roll in Ambient Space + Final Distribution")
```


    
![png](/assets/images/intro-sliced-wasserstein_files/intro-sliced-wasserstein_3_0.png)
    

