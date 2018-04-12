import os
import torch
import torch.utils.data
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
from utils import *
from vae_gan import *

CUDA = False
SEED = 1
BATCH_SIZE = 128
LOG_INTERVAL = 10
EPOCHS = 50
ZDIMS = 20
PDIMS = 30
PRIVATE = False
GAMMA = 25
BETA = 5
# I do this so that the MNIST dataset is downloaded where I want it
os.chdir("/Users/edison/PycharmProjects/vcca_pytorch")

torch.manual_seed(SEED)

# Download or load downloaded MNIST dataset
# shuffle data at every epoch

data1 = load_data('noisymnist_view1.gz')
data2 = load_data('noisymnist_view2.gz')

train_set_x1, _ = data1[0]
train_set_x2, _ = data2[0]

train_loader = torch.utils.data.DataLoader(
             ConcatDataset(
                 train_set_x1,
                 train_set_x2
             ),
             batch_size=BATCH_SIZE, shuffle=True)
print('train_loader')
# Encoder
encoder = Encoder()
# Decoder / Generator
decoder = Decoder()
# Discriminator
discriminator = Discriminator()

optimizer_Enc = optim.Adam(encoder.parameters(), lr=1e-4)
optimizer_Dec = optim.Adam(decoder.parameters(), lr=1e-4)
optimizer_Dis = optim.Adam(discriminator.parameters(), lr=1e-4)


def Auxilary(z):
    z_p = Variable(z.data.new(z.size()).normal_())
    return z_p


def discriminator_loss(recon_data, sample_data, real_data, REAL_LABEL, FAKE_LABEL):
    recon_loss = F.binary_cross_entropy(recon_data, FAKE_LABEL)
    sample_loss = F.binary_cross_entropy(sample_data, FAKE_LABEL)
    real_loss = F.binary_cross_entropy(real_data, REAL_LABEL)

    return recon_loss + sample_loss + real_loss


def train(epoch):
    # toggle model to train mode
    encoder.train()
    decoder.train()
    discriminator.train()
    train_dec_loss = 0
    train_dis_loss = 0
    train_enc_loss = 0
    print('start')
    # in the case of MNIST, len(train_loader.dataset) is 60000
    # each `data` is of BATCH_SIZE samples and has shape [128, 1, 28, 28]
    for batch_idx, (data, _) in enumerate(train_loader):

        data = Variable(data).float()
        optimizer_Enc.zero_grad()
        optimizer_Dec.zero_grad()
        optimizer_Dis.zero_grad()

        # Encoder
        mu, log_var = encoder(data)

        # Decoder / Generator
        z = encoder.reparameterize(mu, log_var)

        # Auxilary Z_p <- samples from prior N(0, I)
        z_p = Auxilary(z)

        x_tilde = decoder.forward(z)
        x_p = decoder(z_p)

        # Discriminator

        X = discriminator(x_tilde)
        X_p = discriminator(x_p)
        X_real = discriminator(data)
        X_sim = discriminator.similarity(x_tilde)
        X_data = discriminator.similarity(data)

        REAL_LABEL = Variable(torch.ones(X_real.size(0)))
        FAKE_LABEL = Variable(torch.zeros(X.size(0)))

        # Discriminator loss
        dis_loss = discriminator_loss(X, X_p, X_real, REAL_LABEL, FAKE_LABEL)

        # Decoder loss
        rec_loss = ((X_sim - X_data) ** 2).mean()
        dec_loss = GAMMA * rec_loss - dis_loss

        # Encoder loss
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        KLD /= BATCH_SIZE * 784
        enc_loss = KLD + BETA * rec_loss

        # Back propagation
        # Train Discriminator
        dis_loss.backward(retain_graph=True)
        optimizer_Dis.step()

        # Train Decoder
        dec_loss.backward(retain_graph=True)
        optimizer_Dec.step()

        # Train Encoder
        enc_loss.backward()
        optimizer_Enc.step()


        train_dec_loss += dec_loss.data[0]
        train_dis_loss += dis_loss.data[0]
        train_enc_loss += enc_loss.data[0]
        #if batch_idx % LOG_INTERVAL == 0:
         #   print('Train Epoch: {} [{}/{} ({:.0f}%)]\tDECLoss: {:.6f} DISLoss: {:.6f} ENCLoss: {:.6f}'.format(
          #      epoch, batch_idx * len(data), len(train_loader.dataset),
           #     100. * batch_idx / len(train_loader),
            #    dec_loss.data[0] / len(data),
             #   dis_loss.data[0] / len(data),
              #  enc_loss.data[0] / len(data)))

    print('====> Epoch: {} Average DECloss: {:.4f} Average DISloss: {:.4f} Average ENCloss: {:.4f}'.format(
          epoch,
        train_dec_loss / len(train_loader.dataset),
        train_dis_loss / len(train_loader.dataset),
        train_enc_loss / len(train_loader.dataset)))



for epoch in range(1, EPOCHS + 1):
    train(epoch)
    #est(epoch)
    decoder.eval()
    # 64 sets of random ZDIMS-float vectors, i.e. 64 locations / MNIST
    # digits in latent space

    sample = Variable(torch.randn(64, ZDIMS))

    sample1 = decoder(sample).cpu()
    # save out as an 8x8 matrix of MNIST digits
    # this will give you a visual idea of how well latent space can generate things
    # that look like digits
    save_image(sample1.data.view(64, 1, 28, 28),
                   '/Users/edison/PycharmProjects/vcca_pytorch/vae_gan/results/sample_' + str(epoch) + '.png')

with open('encoder.pt','wb') as f:
    torch.save(encoder, f)
with open('decoder.pt','wb') as f:
    torch.save(decoder, f)
with open('discriminator.pt','wb') as f:
    torch.save(discriminator, f)