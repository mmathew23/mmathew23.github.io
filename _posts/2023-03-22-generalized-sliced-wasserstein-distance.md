---
title: Generalized Sliced Wasserstein Distance
date: 2023-03-22
categories: [AI, Sliced Wasserstein]
# layout: posts
tags: [AI, Sliced Wasserstein, MNIST]
---

The data we model in machine learning tends to reside on a lower-dimensional sub-manifold, [Manifold Hypothesis](https://en.wikipedia.org/wiki/Manifold_hypothesis) . On the one hand, Sliced Wasserstein Distance with enough projection directions can capture the inherent nonlinear structure in the data. On the other hand, Max Sliced Wasserstein uses only one linear projection direction, which is good for memory but cannot capture the data's topology in one pass. A logical next step is to learn a subspace that maximally separates two distributions.

There are three critical components to Sliced Wasserstein: the projection parameters residing on the unit hypersphere, the projection operator, and the one-dimensional projection sorting. The projection operator is the inner product between vectors, a linear operator. Let's instead consider using a nonlinear operator, which is parameterized by $$\theta$$, real-valued, homogeneous of degree one for the projection parameters $$\theta$$, non-degenerate, and strictly positive Hessian. Let's also relax the assumption that the projection parameters lie on the unit hypersphere. Replacing the linear operator and integration over all unit hyperspheres with a nonlinear operator and integration over a compact set of feasible parameters leads to the definition of Generalized Sliced Wasserstein Distance (GSWD).

$$ \boldsymbol{GSW}_p^p = \int_{\Omega_{\theta}} \boldsymbol{W}_p^p(\mathcal{G}\mathcal{I}_{\mu}(\cdot, \theta), \mathcal{G}\mathcal{I}_{\nu}(\cdot, \theta)) d\theta$$

where $$\mathcal{G}\mathcal{I}$$ is the distribution projected by the nonlinear operator [S. Kolouri](https://arxiv.org/abs/1902.00434).
Like max SWD, we can also empirically maximize the distance between distributions with GSWD by optimizing $$\theta$$, called max GSWD. Choosing an appropriate function remains an open question, but the linked paper above further clarifies it must be injective. It lists three choices: the circular defining function, homogeneous polynomials with an odd degree, and neural networks with an appropriate activation (Leaky ReLU or ReLU).

GSWD still needs to choose the number of projection directions. Moreover, considering the nonlinearity of the projection operator, it could require even more projections to capture the complete topology of the data. In the paper, the authors note the choice of projection function should be data-dependent but that the polynomials of degree 3 or 5 tend to perform well across all tasks. However, the drawback of using polynomials is the projection complexity increases exponentially in data dimension and degree of the polynomial, which means it's impractical for large-scale ML.

Here's an example of improving SWD using GSWD for MNIST digit generation. The core idea is train an MNIST autoencoder, use the encoder as our generatlized function and learn to sample the latent space. Once done we can sample from the latent space and use the decoder part of the autoencoder to generate MNIST samples.

```python
class MnistEncoder(nn.Module):
    def __init__(self, final_dim):
        super(MnistEncoder, self).__init__()
        self.final_dim = final_dim

        self.convs = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(12, 24, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(24, 49, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(49, 49, kernel_size=4, padding=0),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(49, 49, kernel_size=3, padding=1),
        )

        self.linear = nn.Sequential(
            nn.Linear(49*16, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, self.final_dim)
        )

    def forward(self, x):
        return self.linear(self.convs(x).flatten(1))


class MnistDecoder(nn.Module):
    def __init__(self, initial_dim):
        super(MnistDecoder, self).__init__()
        self.initial_dim = initial_dim
        self.linear = nn.Sequential(
            nn.Linear(self.initial_dim, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 49*16)
        )

        upsample = nn.UpsamplingNearest2d
        self.convs = nn.Sequential(
            nn.Conv2d(49, 49, kernel_size=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2),
            upsample(scale_factor=2),
            nn.Conv2d(49, 49, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(49, 49, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            upsample(scale_factor=2),
            nn.Conv2d(49, 25, kernel_size=3, padding=0),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(25, 25, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            upsample(scale_factor=2),
            nn.Conv2d(25, 25, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(25, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.convs(self.linear(x).view(x.shape[0], -1, 4, 4))

class LatentGenerator(nn.Module):
    def __init__(self, dim):
        super(LatentGenerator, self).__init__()
        self.dim = dim
        self.linear_layers = nn.Sequential(
            nn.Linear(self.dim//4, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, self.dim),
        )

    def forward(self, batch_size=1):
        device = self.linear_layers[0].weight.data.device
        z = torch.randn(batch_size, self.dim//4, device=device)
        return self.linear_layers(z)
```

Assuming the autoencoder is trained let's look at the training loop for the `LatentGenerator`.

```python
    for epoch in range(epochs):
        for iteration, (image, _) in enumerate(dataloader):
            image = image.cuda(device)
            optimizer.zero_grad()
            z = generator(image.shape[0])
            with torch.no_grad():
                target_z = ae.encoder(image)
            loss = sw_loss(z.flatten(1), target_z.flatten(1))
            loss.backward()
            optimizer.step()
            scheduler.step()
```

`sw_loss` takes the encoded images, which comes from a neural network, then randomly projects onto a linear subspace. Regular Sliced Wasserstein would skip the neural network step entirely, but it proves to be critical to get quality results. Below are uncurated generated samples from the generator and decoded into images using the decoder. Full training implementation can be found on [github](https://github.com/mmathew23/sliced_wasserstein) in the `train_mnist.py` file.

{% include figure image_path="/assets/images/generalized-sliced-wasserstein-distance_files/mnist_generated.png" alt="This is a picture of generated digit samples based on the MNIST dataset. It is arranged as a 20 by 20 grid with one digit occupying each grid location" caption="400 MNIST Samples" %}

There is still a key difference in my formulation vs the theorized GSWD. In theory we don't need a trained neural network. We could just use random weights. The problem with this approach is that the space to sample directions from grows even quicker than a plain linear space, which defeats the purpose of efficiently sampling to acheive quality results. That said, armed with a feature extractor, we can employ GSWD to train a data generator quickly.



