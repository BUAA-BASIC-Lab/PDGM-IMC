import torch
from torch import nn
from nflows.transforms.base import Transform
from nflows.nn.nets.resnet import ResidualBlock
from torch.nn import functional as F
from utils import denormalize
import numpy as np


class Physical_Layer(Transform):
    """
    A custom Physical-Driven Normalizing Flow Layer that incorporates non-ideality priors derived from device- and circuit-level formulations.

    Mean formula:
        mean = term1_mean * term2_mean * (1 / (127 * 255)) + term3
        where:
            term1_mean = x * (x - 128) * w
            term2_mean = 1 / (a * |w| + b) - c
            term3_mean = d * x * (w / |w|)
        *Note: Term3 is a correction specific to our experimental chip and can be removed for general analog in-memory computing circuits.

    Std formula:
        std = sqrt[term1_std ^ 2 + term2_std ^ 2]
        where:    
            term1_std = |w| * (alpha_1 * sqrt(x^2+x) + alpha_2)
            term2_std = x * (beta_1 * |w| + beta_2)

    If `die_encode_net` is provided, it adjusts mean and std using one-hot encoded die context.
        adjusted_mean = k_mu * mean + b_mu
        adjusted_std = k_sigma * std + b_sigma
        where:
            k_mu, b_mu, k_sigma, b_sigma = die_encode_net(one_hot(die_id))

    Args:
        features (int): Dimensionality of the noise vector u to be sampled.
        x_dim (int): Dimensionality of the input x to the compute-in-memory array.
        w_dim (int): Dimensionality of the weight w in the compute-in-memory array.
                     Must equal `features`.
        norm_params (dict): Dictionary with min/max values for x and w for denormalization.

    Optional Args:
        n_die (int, optional): Number of die types (used for die one-hot encoding).
                               Required if `die_encode_net` is used.
        a (float, optional): Device nonlinearity parameter (default: 0.11).
        b (float, optional): Device nonlinearity parameter (default: 0.29).
        c (float, optional): Device nonlinearity parameter (default: 0.18).
        d (float, optional): Device nonlinearity parameter (default: 0.04).
        alpha_1 (float, optional): Scaling factor for DAC noise (default: 1e-5).
        alpha_2 (float, optional): Bias for DAC noise (default: 0.03).
        beta_1 (float, optional): Scaling factor for programming variation (default: 3.7e-4).
        beta_2 (float, optional): Bias for programming variation (default: 0.013).
        trainable (bool, optional): Whether the parameters a, b, c, d, alpha, beta
                                    are learnable during training (default: True).
        die_encode_net (nn.Module, optional): A neural network that maps die one-hot
                                              encoding to adjustment parameters
                                              (k_mu, b_mu, k_sigma, b_sigma).
        max_rescale (float, optional): Maximum allowed rescaling factor for die-based
                                       adjustment (default: 10.0).
    """

    def __init__(self, features, x_dim, w_dim, norm_params, n_die=None,
                 a=0.11, b=0.29, c=0.18, d=0.04,
                 alpha_1=1e-5, alpha_2=0.03, beta_1=3.7e-4, beta_2=0.013,
                 trainable=True, die_encode_net=None, max_rescale=10):
        super().__init__()
        self.features = features
        self.x_dim = x_dim
        self.w_dim = w_dim
        self.n_die = n_die
        self.norm_params = norm_params
        self.die_encode_net = die_encode_net
        self.max_rescale = max_rescale
        self.eps = 1e-6

        assert w_dim == features, "w_dim must match input feature dimension"

        self.context_features = x_dim + w_dim + (n_die if die_encode_net else 0)

        # Mean parameters
        self.a = nn.Parameter(torch.tensor(a), requires_grad=trainable)
        self.b = nn.Parameter(torch.tensor(b), requires_grad=trainable)
        self.c = nn.Parameter(torch.tensor(c), requires_grad=trainable)
        self.d = nn.Parameter(torch.tensor(d), requires_grad=trainable)

        # Std parameters (log space to ensure positivity)
        self.log_alpha_1 = nn.Parameter(torch.log(torch.tensor(abs(alpha_1))), requires_grad=trainable)
        self.log_alpha_2 = nn.Parameter(torch.log(torch.tensor(abs(alpha_2))), requires_grad=trainable)
        self.log_beta_1 = nn.Parameter(torch.log(torch.tensor(abs(beta_1))), requires_grad=trainable)
        self.log_beta_2 = nn.Parameter(torch.log(torch.tensor(abs(beta_2))), requires_grad=trainable)

    @property
    def alpha_1(self): return torch.exp(self.log_alpha_1)
    @property
    def alpha_2(self): return torch.exp(self.log_alpha_2)
    @property
    def beta_1(self): return torch.exp(self.log_beta_1)
    @property
    def beta_2(self): return torch.exp(self.log_beta_2)

    def get_adjust_param(self, encoded_die):
        output = self.die_encode_net(encoded_die)
        k_sigma = torch.sigmoid(output[:, 0:1]) * self.max_rescale
        b_sigma = output[:, 1:2]
        k_mu = torch.sigmoid(output[:, 2:3]) * self.max_rescale
        b_mu = output[:, 3:4]
        return k_mu, b_mu, k_sigma, b_sigma

    def inverse(self, inputs, context):

        batch_size, input_dim = inputs.shape
        assert input_dim == self.features
        assert context.shape[1] == self.context_features

        x_norm = context[:, :self.x_dim]
        w_norm = context[:, self.x_dim:self.x_dim + self.w_dim]
        die_tensor = context[:, self.x_dim + self.w_dim:] if self.die_encode_net else None

        x, w = denormalize(x_norm, w_norm, self.norm_params)

        x_exp = self._expand_x_to_match_dim(x, input_dim) if self.x_dim != input_dim else x

        term1_mean = x_exp * (x_exp - 128) * w
        term2_mean = 1 / (self.a * torch.abs(w) + self.b) - self.c
        term3_mean = self.d * x_exp * (w / (torch.abs(w) + self.eps)) # This term is a correction specific to our experimental chip and can be removed for general analog in-memory computing circuits.
        mean = term1_mean * term2_mean * (1 / (127 * 255)) + term3_mean

        term1_std = torch.abs(w) * (self.alpha_1 * torch.sqrt(x_exp ** 2 + x_exp) + self.alpha_2)
        term2_std = x_exp * (self.beta_1 * torch.abs(w) + self.beta_2)
        std = torch.sqrt(term1_std ** 2 + term2_std ** 2 + self.eps)

        if die_tensor is not None:
            k_mu, b_mu, k_sigma, b_sigma = self.get_adjust_param(die_tensor)
            mean = mean * k_mu.expand_as(mean) + b_mu.expand_as(mean)
            std = std * k_sigma.expand_as(std) + b_sigma.expand_as(std)

        outputs = inputs * std + mean
        logabsdet = torch.sum(torch.log(torch.abs(std) + self.eps), dim=1)
        return outputs, logabsdet

    def forward(self, inputs, context):

        batch_size, input_dim = inputs.shape
        assert input_dim == self.features
        assert context.shape[1] == self.context_features

        x_norm = context[:, :self.x_dim]
        w_norm = context[:, self.x_dim:self.x_dim + self.w_dim]
        die_tensor = context[:, self.x_dim + self.w_dim:] if self.die_encode_net else None

        x, w = denormalize(x_norm, w_norm, self.norm_params)

        x_exp = self._expand_x_to_match_dim(x, input_dim) if self.x_dim != input_dim else x

        term1_mean = x_exp * (x_exp - 128) * w
        term2_mean = 1 / (self.a * torch.abs(w) + self.b) - self.c
        term3_mean = self.d * x_exp * (w / (torch.abs(w) + self.eps)) # This term is a correction specific to our experimental chip and can be removed for general analog in-memory computing circuits.
        mean = term1_mean * term2_mean * (1 / (127 * 255)) + term3_mean

        term1_std = torch.abs(w) * (self.alpha_1 * torch.sqrt(x_exp ** 2 + x_exp) + self.alpha_2)
        term2_std = x_exp * (self.beta_1 * torch.abs(w) + self.beta_2)
        std = torch.sqrt(term1_std ** 2 + term2_std ** 2 + self.eps)

        if die_tensor is not None:
            k_mu, b_mu, k_sigma, b_sigma = self.get_adjust_param(die_tensor)
            mean = mean * k_mu.expand_as(mean) + b_mu.expand_as(mean)
            std = std * k_sigma.expand_as(std) + b_sigma.expand_as(std)

        outputs = (inputs - mean) / (std + self.eps)
        logabsdet = -torch.sum(torch.log(torch.abs(std) + self.eps), dim=1)
        return outputs, logabsdet

    def _expand_x_to_match_dim(self, x, target_dim):
        batch_size, x_dim = x.shape
        if x_dim == 1:
            return x.repeat(1, target_dim)
        repeats = torch.tensor([target_dim // x_dim] * x_dim)
        repeats[:target_dim % x_dim] += 1
        return torch.cat([x[:, i:i+1].repeat(1, r) for i, r in enumerate(repeats)], dim=1)
    
    

class ResidualNet(nn.Module):

    # Adapted from:
    # Conor Durkan, Artur Bekasov, Iain Murray, & George Papamakarios. (2020).
    # nflows: normalizing flows in PyTorch (v0.14). Zenodo.
    # https://doi.org/10.5281/zenodo.4296287

    # Original implementation:
    # https://github.com/bayesiains/nflows

    # Modifications:
    # - Added `zero_initialization` for identity-like behavior at init


    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        zero_initialization=False,  # New param
    ):
        super().__init__()
        self.hidden_features = hidden_features
        self.context_features = context_features
        if context_features is not None:
            self.initial_layer = nn.Linear(
                in_features + context_features, hidden_features
            )
        else:
            self.initial_layer = nn.Linear(in_features, hidden_features)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    features=hidden_features,
                    context_features=context_features,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layer = nn.Linear(hidden_features, out_features)
        
        # If initialization as an identity transform is required, the final layer is initialized to zero.
        if zero_initialization:
            nn.init.zeros_(self.final_layer.weight)
            nn.init.zeros_(self.final_layer.bias)

    def forward(self, inputs, context=None):
        if context is None:
            temps = self.initial_layer(inputs)
        else:
            temps = self.initial_layer(torch.cat((inputs, context), dim=1))
        for block in self.blocks:
            temps = block(temps, context=context)
        outputs = self.final_layer(temps)
        return outputs
    
class Output_Layer(torch.nn.Module):
    """
    Non-ideality Model for Output Circuit.

    Models the conditional distribution p(error | y) as a Laplace distribution
    with:
        mean(y)  = poly3(y; coeffs)
        sigma(y) = log(k * |y| + b)

    Args:
        device (str): Device to store parameters ('cpu' or 'cuda').
        trainable (bool): If True, parameters are updated during training.
        init_mean_coeffs (tuple of float): Initial cubic polynomial coefficients
            [c3, c2, c1] for the conditional mean μ(y).
        init_std_scale (float): Initial k parameter in the log-linear scale function.
        init_std_bias (float): Initial b parameter in the log-linear scale function.

    Laplace distribution is sharper at the peak and has heavier tails than Gaussian
    """
    def __init__(
        self,
        device='cpu',
        trainable=True,
        init_mean_coeffs=(8.46e-5, 3.97e-3, -6.68e-2), #These parameters are pretrained.
        init_std_scale=1.51, 
        init_std_bias=8.06   
    ):
        super().__init__()
        self.distribution = 'laplace'

        # Polynomial coefficients for the conditional mean μ(y)
        initial_coeffs = torch.tensor(init_mean_coeffs, dtype=torch.float32, device=device)
        self.coeffs = torch.nn.Parameter(initial_coeffs, requires_grad=trainable)

        # log(k) and log(b) for the conditional scale function
        self.log_k = torch.nn.Parameter(
            torch.tensor(np.log(init_std_scale), dtype=torch.float32, device=device),
            requires_grad=trainable
        )
        self.log_b = torch.nn.Parameter(
            torch.tensor(np.log(init_std_bias), dtype=torch.float32, device=device),
            requires_grad=trainable
        )

    def forward(self, y):
        powers = torch.stack([y ** i for i in reversed(range(3))], dim=-1)
        mean = (powers * self.coeffs).sum(dim=-1)
        k = torch.exp(self.log_k)
        b = torch.exp(self.log_b)
        sigma = torch.log(k * torch.abs(y) + b + 1e-6)  # log-linear scale modeling
        return mean, sigma

    def negative_log_likelihood(self, mean, sigma, error):
        scale = sigma / np.sqrt(2)
        return torch.log(2 * scale) + torch.abs(error - mean) / scale



if __name__ == "__main__":
    import torch
    from torch import nn

    torch.manual_seed(42)

    # Normalization settings
    norm_params = {
        'x_min': 0.0, 'x_max': 255.0,
        'w_min': -128.0, 'w_max': 128.0
    }

    # Example die adjustment network
    class DieAdjustNet(nn.Module):
        def __init__(self, n_die, max_rescale=10):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(n_die, 64),
                nn.ReLU(),
                nn.Linear(64, 4)
            )
            self.max_rescale = max_rescale

        def forward(self, x):
            out = self.network(x)
            k_mu = torch.sigmoid(out[:, 0:1]) * self.max_rescale
            b_mu = out[:, 1:2]
            k_sigma = torch.sigmoid(out[:, 2:3]) * self.max_rescale
            b_sigma = out[:, 3:4]
            return torch.cat([k_mu, b_mu, k_sigma, b_sigma], dim=1)

    # Model setup
    n_die = 5
    features = 4
    x_dim = 2
    w_dim = features

    die_net = DieAdjustNet(n_die=n_die)
    layer = Physical_Layer(
        features=features,
        x_dim=x_dim,
        w_dim=w_dim,
        norm_params=norm_params,
        n_die=n_die,
        die_encode_net=die_net
    )

    # Dummy inputs
    batch_size = 3
    u = torch.randn(batch_size, features)
    x_norm = torch.rand(batch_size, x_dim)
    w_norm = torch.rand(batch_size, w_dim) * 2 - 1
    die_tensor = torch.eye(n_die)[torch.tensor([1, 3, 0])]

    context = torch.cat([x_norm, w_norm, die_tensor], dim=1)

    with torch.no_grad():
        # Forward: u → z
        z, logdet = layer.forward(u, context)
        print("Forward z:\n", z)
        print("Log|det| (forward):\n", logdet)

        # Inverse: z → u
        u_rec, logdet_inv = layer.inverse(z, context)
        print("Reconstructed u:\n", u_rec)
        print("Log|det| (inverse):\n", logdet_inv)

        # Reconstruction error
        recon_error = torch.norm(u - u_rec).item()
        print(f"Reconstruction error: {recon_error:.6e}")
        assert recon_error < 1e-4, "Reconstruction error too large!"

        # Check learned parameters
        print(f"alpha_1: {layer.alpha_1.item():.2e}")
        print(f"alpha_2: {layer.alpha_2.item():.2e}")
        print(f"beta_1 : {layer.beta_1.item():.2e}")
        print(f"beta_2 : {layer.beta_2.item():.2e}")

