import torch
from torch import nn
from torch.autograd import Variable

ZDIMS = 100


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=5,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=5,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=5,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.fn1 = nn.Sequential(
            nn.Linear(256 * 5 * 5, 2048),
            nn.LeakyReLU(0.2),
        )
        self.mu = nn.Sequential(
            nn.Linear(2048, 100),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.logvar = nn.Sequential(
            nn.Linear(2048, 100),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 256 * 5 * 5)
        x = self.fn1(x)
        return self.mu(x), self.logvar(x)

    def reparameterize(self, mu, logvar):
        """THE REPARAMETERIZATION IDEA:

        """

        if self.training:

            std = logvar.mul(0.5).exp_()  # type: Variable

            eps = Variable(std.data.new(std.size()).normal_()).cuda()

            return eps.mul(std).add_(mu)

        else:
            # During inference, we simply spit out the mean of the
            # learned distribution for the current input.  We could
            # use a random sample from the distribution, but mu of
            # course has the highest probability.
            return mu


class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.linear = torch.nn.Linear(100, 8 * 8 * 256)

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=256, kernel_size=5,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=5,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128, out_channels=32, kernel_size=5,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32, out_channels=3, kernel_size=5,
                stride=2, padding=1, bias=False
            )
        )
        self.out = torch.nn.Tanh()

    def forward(self, x):
        # Project and reshape
        x = self.linear(x)
        x = x.view(x.shape[0], 256, 8, 8)
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Apply Tanh
        return self.out(x)


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=5,
                stride=2, padding=1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=128, kernel_size=5,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=5,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=5,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fn1 = nn.Sequential(
            nn.Linear(28 * 28 * 1024 * 2, 512 * 5 * 5),
            nn.LeakyReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(512 * 5 * 5, 1),
            nn.Sigmoid(),
        )
        self.out2 = nn.Linear(512 * 5 * 5, 1)

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Flatten and apply sigmoid
        x = x.view(-1, 28 * 28 * 1024 * 2)
        x = self.fn1(x)
        x = self.out(x)
        return x

    def similarity(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Flatten and apply sigmoid
        x = x.view(-1, 28 * 28 * 1024 * 2)
        x = self.fn1(x)
        x = self.out2(x)
        return x
