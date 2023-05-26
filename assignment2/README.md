# Assignment 2: Convolutional Neural Networks with Pytorch

This notebook has 5 parts. We will learn PyTorch on **three different levels of abstraction**

1. Part I, Preparation: we will use CIFAR-100 dataset.
2. Part II, Barebones PyTorch: **Abstraction level 1**, we will work directly with the lowest-level PyTorch Tensors. 
3. Part III, PyTorch Module API: **Abstraction level 2**, we will use `nn.Module` to define arbitrary neural network architecture. 
4. Part IV, PyTorch Sequential API: **Abstraction level 3**, we will use `nn.Sequential` to define a linear feed-forward network very conveniently. 
5. Part V. ResNet10 Implementation: we will implement ResNet10 from scratch given the architecture details
5. Part VI, CIFAR-100 open-ended challenge: we will implement our own network to get as high accuracy as possible on CIFAR-100. We can experiment with any layer, optimizer, hyperparameters or other advanced features. 

Here is a table of comparison:

| API           | Flexibility | Convenience |
|---------------|-------------|-------------|
| Barebone      | High        | Low         |
| `nn.Module`     | High        | Medium      |
| `nn.Sequential` | Low         | High        |
