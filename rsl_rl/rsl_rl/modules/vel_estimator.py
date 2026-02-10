import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class VelEstimator(nn.Module):
    """Velocity Estimator using simple regression."""
    def __init__(self,
                 temporal_steps,
                 num_one_step_obs,
                 enc_hidden_dims=[128, 64],
                 activation='elu',
                 learning_rate=1e-3,
                 max_grad_norm=10.0,
                 **kwargs):
        if kwargs:
            print("VelEstimator.__init__ got unexpected arguments: " + str([key for key in kwargs.keys()]))
        super(VelEstimator, self).__init__()
        activation = get_activation(activation)

        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.max_grad_norm = max_grad_norm

        # Encoder - only outputs velocity (3 dims)
        enc_input_dim = self.temporal_steps * self.num_one_step_obs
        enc_layers = []
        for l in range(len(enc_hidden_dims)):
            enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[l]), activation]
            enc_input_dim = enc_hidden_dims[l]
        enc_layers += [nn.Linear(enc_input_dim, 3)]  # Output 3D velocity
        self.encoder = nn.Sequential(*enc_layers)

        # Optimizer
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, obs_history):
        vel = self.encoder(obs_history.detach())
        return vel.detach()

    def encode(self, obs_history):
        vel = self.encoder(obs_history.detach())
        return vel

    def update(self, obs_history, next_critic_obs, lr=None):
        if lr is not None:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
                
        # Extract ground truth velocity from privileged observations
        vel_gt = next_critic_obs[:, self.num_one_step_obs:self.num_one_step_obs+3].detach()

        # Predict velocity from history
        pred_vel = self.encoder(obs_history)

        # Simple MSE loss for velocity regression
        loss = F.mse_loss(pred_vel, vel_gt)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return loss.item()


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "silu":
        return nn.SiLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
