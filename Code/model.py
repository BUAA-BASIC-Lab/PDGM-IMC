import torch
from Code.Open_Source.Src.layers import ResidualNet, Physical_Layer
from nflows import flows, transforms, distributions, utils
import os

def build_CNF(norm_params,
            condition_dim=135,
            target_dim=128,
            n_die=3,
            num_transforms=8,
            hidden_features=512,
            max_rescale=2.0,

            a=0.11, b=0.29, c=0.18, d=0.04,
            alpha_1=1e-5, alpha_2=0.03,
            beta_1=3.7e-4, beta_2=0.013,
            trainable=True,

            die_encode_net=None,
            transform_net_create_fn=None):
    """
    Builds a Conditional Normalizing Flow in PDGM-IMC as a nflows.Flow object.

    Args:
        norm_params (dict): Min/max ranges for x and w (for normalization/denormalization).
        condition_dim (int): Total dimension of conditioning variables (x + w + die).
        target_dim (int): Dimension of the target variable (e.g., noise).
        n_die (int): Number of die types (used for one-hot encoding).
        num_transforms (int): Number of normalizing flow transformation blocks.
        hidden_features (int): Hidden layer size in neural nets.
        max_rescale (float): Max scaling factor for die-based mean/std correction.

        a, b, c, d (float): Parameters for device nonlinearity modeling.
        alpha_1, alpha_2 (float): Parameters for DAC noise modeling.
        beta_1, beta_2 (float): Parameters for programming noise modeling.
        trainable (bool): Whether the physical parameters are learnable.

        die_encode_net (nn.Module): Optional network to adjust mean/std via multiple dies.
        transform_net_create_fn (callable): Function to create networks for affine coupling.
                                            Signature: (in_features, out_features) -> nn.Module

    Returns:
        flows.Flow: The constructed normalizing flow model.
    """

    x_dim = condition_dim - target_dim - n_die
    w_dim = target_dim

    transform_list = []

    # Default die encoder if not provided
    if die_encode_net is None:
        die_encode_net = ResidualNet(
            in_features=n_die,
            out_features=4,
            hidden_features=hidden_features,
            num_blocks=4,
            activation=torch.nn.ReLU(),
            zero_initialization=True
        )

    transform_list.append(
        Physical_Layer(
            features=target_dim,
            x_dim=x_dim,
            w_dim=w_dim,
            n_die=n_die,
            norm_params=norm_params,
            a=a, b=b, c=c, d=d,
            alpha_1=alpha_1, alpha_2=alpha_2,
            beta_1=beta_1, beta_2=beta_2,
            trainable=trainable,
            die_encode_net=die_encode_net,
            max_rescale=max_rescale
        )
    )

    # Default transform net for coupling layer
    if transform_net_create_fn is None:
        transform_net_create_fn = lambda in_features, out_features: ResidualNet(
            in_features=in_features,
            out_features=out_features,
            context_features=condition_dim,
            hidden_features=hidden_features,
            num_blocks=4,
            activation=torch.nn.ReLU(),
            zero_initialization=True
        )

    for _ in range(num_transforms):
        transform_list.append(transforms.ActNorm(features=target_dim))

        transform_list.append(
            transforms.AffineCouplingTransform(
                mask=utils.create_random_binary_mask(features=target_dim),
                transform_net_create_fn=transform_net_create_fn
            )
        )

        transform_list.append(transforms.LULinear(features=target_dim, identity_init=True))

    return flows.Flow(
        transform=transforms.CompositeTransform(transform_list),
        distribution=distributions.StandardNormal([target_dim])
    )

def save_model(model, optimizer, epoch, loss, norm_params, model_save_dir, filename):
    os.makedirs(model_save_dir, exist_ok=True)
    model_path = os.path.join(model_save_dir, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'norm_params': norm_params
    }, model_path)
    print(f"Model saved to {model_path}")


def load_cnf_model(model_path, model_args, device="cpu"):
    """
    Load CNF model from checkpoint using build_CNF() with user-provided arguments.

    Args:
        model_path (str): Path to saved checkpoint (.pt file).
        model_args (dict): Arguments passed directly to build_CNF().
        device (str or torch.device): Target device.

    Returns:
        model, norm_params
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint {model_path} not found.")

    checkpoint = torch.load(model_path, map_location=device)
    norm_params = checkpoint["norm_params"]
    model_args["norm_params"] = norm_params

    model = build_CNF(**model_args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, norm_params





if __name__ == "__main__":
    import torch

    torch.manual_seed(0)

    # Normalization configuration (example chip settings)
    norm_params = {
        'x_min': 0.0, 'x_max': 255.0,
        'w_min': -128.0, 'w_max': 128.0
    }

    # Dimensions
    batch_size = 4
    x_dim = 2
    w_dim = 4
    n_die = 5
    condition_dim = x_dim + w_dim + n_die
    target_dim = w_dim

    # Create normalized inputs
    x_norm = torch.rand(batch_size, x_dim)
    w_norm = torch.rand(batch_size, w_dim) * 2 - 1  # in [-1, 1]
    die_tensor = torch.eye(n_die)[torch.tensor([0, 2, 3, 4])]
    context = torch.cat([x_norm, w_norm, die_tensor], dim=1)

    # Build the PDGM-IMC model (flow)
    model = build_CNF(
        norm_params=norm_params,
        condition_dim=condition_dim,
        target_dim=target_dim,
        n_die=n_die,
        num_transforms=4,
        hidden_features=64
    )

    u = torch.randn(batch_size, target_dim)

    # Evaluate log-likelihood of inputs
    log_prob = model.log_prob(u, context)
    print("Log-likelihood:\n", log_prob)

    # Sample from the model
    samples = model.sample(3, context[:3])
    print("Generated samples:\n", samples)

