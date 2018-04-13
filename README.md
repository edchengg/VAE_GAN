# VAE/GAN

This is an Pytorch implementation of [Autoencoding beyond pixels using a learned similarity metric(VAE/GAN)](https://arxiv.org/abs/1512.09300) in Python.

## Variational Autoencoder + Generative Adverisal Network

Encoder + Decoder/Generator + Discriminator

<img src="https://github.com/edchengg/VAE_GAN/blob/master/imgs/vaegan.png" width="300">

## VAE/GAN in training

Encoder + Decoder/Generator + Discriminator

<img src="https://github.com/edchengg/VAE_GAN/blob/master/imgs/vaegant.png" width="300">

## Training Algorithm

<img src="https://github.com/edchengg/VAE_GAN/blob/master/imgs/algo.png" width="300">

### similarity

```python
class Discriminator
	self.f4 = nn.Linear(256, 1)

	def similarity(self, x):
	....
	return self.f4(self.dropout(h)) # no sigmoid at the end
```

## Loss Function in python code

### Decoder/Generator Loss Function
```python
X_sim = discriminator.similarity(x_tilde)
X_data = discriminator.similarity(data)
rec_loss = ((X_sim - X_data) ** 2).mean()
dec_loss = GAMMA * rec_loss - dis_loss
```

### Encoder Loss Function
```python
KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
KLD /= BATCH_SIZE * 784
enc_loss = KLD + BETA * rec_loss
```

### Discriminator Loss Function
```python
recon_loss = F.binary_cross_entropy(recon_data, FAKE_LABEL)
sample_loss = F.binary_cross_entropy(sample_data, FAKE_LABEL)
real_loss = F.binary_cross_entropy(real_data, REAL_LABEL)
dis_loss = recon_loss + sample_loss + real_loss
```

### Data
MNIST

### Neural Network

| Encoder       | Decoder          | Discriminator  |
| :-------------: |:-------------:| :-----:|
| 784 * 1024, LeakyReLU | 20 * 1024, LeakyReLU | 784 * 1024, LeakyReLU  |
| 1024 * 1024, LeakyReLU | 1024 * 1024, LeakyReLU | 1024 * 512, LeakyReLU  |
| 1024 * 1024, LeakyReLU | 1024 * 1024, LeakyReLU | 512 * 256, LeakyReLU |
| 1024 * 20 | 1024 * 784, Tanh | 256 * 1, Sigmoid |

### Generation results
Loss:
<img src="https://github.com/edchengg/VAE_GAN/blob/master/imgs/Figure_1.png" width="300">
<img src="https://github.com/edchengg/VAE_GAN/blob/master/imgs/Figure_2.png" width="300">
1-50 epochs:

![Alt Text](https://github.com/edchengg/VAE_GAN/blob/master/imgs/result.gif)