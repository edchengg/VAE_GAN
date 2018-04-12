import torch
from torch import nn
from torch.autograd import Variable
ZDIMS=20


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.DIMS = 20
        self.dropout = nn.Dropout(0.10)
        self.relu = nn.LeakyReLU(0.2)
        # ENCODER 1
        # 28 x 28 pixels = 784 input pixels, 400 outputs
        self.en_x_1 = nn.Linear(784, 1024)
        self.en_x_2 = nn.Linear(1024, 1024)
        self.en_x_3 = nn.Linear(1024, 1024)
        self.en_x_4_mu = nn.Linear(1024, ZDIMS)  # mu layer
        self.en_x_4_sigma = nn.Linear(1024,ZDIMS)  # log vairiance

    def forward(self, x):
        """Input vector x -> fully connected 1 -> ReLU -> (fully connected

        """
        h = self.relu(self.en_x_1(self.dropout(x)))
        h = self.relu(self.en_x_2(self.dropout(h)))
        h = self.relu(self.en_x_3(self.dropout(h)))
        return self.en_x_4_mu(self.dropout(h)), self.en_x_4_sigma(self.dropout(h))

    def reparameterize(self, mu, logvar):
        """THE REPARAMETERIZATION IDEA:

        """

        if self.training:

            std = logvar.mul(0.5).exp_()  # type: Variable

            eps = Variable(std.data.new(std.size()).normal_())

            return eps.mul(std).add_(mu)

        else:
            # During inference, we simply spit out the mean of the
            # learned distribution for the current input.  We could
            # use a random sample from the distribution, but mu of
            # course has the highest probability.
            return mu



class Decoder(nn.Module):
    def __init__(self, ZDIMS=20):
        super().__init__()


        self.dropout = nn.Dropout(0.10)
        self.relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()
        # DECODER

        self.de_x_1 = nn.Linear(ZDIMS, 1024)
        self.de_x_2 = nn.Linear(1024, 1024)
        self.de_x_3 = nn.Linear(1024, 1024)
        self.de_x_4 = nn.Linear(1024, 784)


    def forward(self, z: Variable) -> Variable:
        h = self.relu(self.de_x_1(self.dropout(z)))
        h = self.relu(self.de_x_2(self.dropout(h)))
        h = self.relu(self.de_x_3(self.dropout(h)))
        return self.tanh(self.de_x_4(self.dropout(h)))



class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
        self.f1 = nn.Linear(784, 1024)
        self.f2 = nn.Linear(1024, 512)
        self.f3 = nn.Linear(512, 256)
        self.f4 = nn.Linear(256, 1)

    def forward(self, x):
        h = self.leakyrelu(self.f1(self.dropout(x)))
        h = self.leakyrelu(self.f2(self.dropout(h)))
        h = self.leakyrelu(self.f3(self.dropout(h)))
        return self.sigmoid(self.f4(self.dropout(h)))

    def similarity(self, x):
        h = self.leakyrelu(self.f1(self.dropout(x)))
        h = self.leakyrelu(self.f2(self.dropout(h)))
        h = self.leakyrelu(self.f3(self.dropout(h)))
        return self.f4(self.dropout(h))

