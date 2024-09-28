### Evaluating How Differential Privacy Can Defense Reconstruction Tasks (Class project for CSE 5479 at The Ohio State University)

This code reproduces the reconstruction attack described in [Hayes et. al.](https://arxiv.org/abs/2302.07225). It tests how adding differential privacy during training neural networks based on the dataset of MNIST can prevent attacker from exploiting reconstruction attack on the neural network. This uses the opacus library for implementing differential pricavy. The code for training MNIST is based on the [opacus example](https://github.com/pytorch/opacus/blob/main/examples/mnist.py). It considered two kinds of reconstruction attacks: model-based and gradient-based.

To run a model-based attack experiment evaluating on both a non private neural network and a private neural network.

```
python3 mnist dp.py --reco_target 0
```

The `reco_target` means the attack algorithm tries to reconstruct the data point indexed at 0 in the MNIST dataset.

To run a gradient-based attack experiment on a private neural network.

```
python3 mnist dp.py --gradient based attack --reco target 0
```

And to run a gradient-based attack experiment on a non-private neural network.

```
python3 mnist dp.py --gradient based attack --reco target 0 --disable-dp
```

