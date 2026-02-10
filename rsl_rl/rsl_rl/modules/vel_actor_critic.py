import torch
import torch.nn as nn
from torch.distributions import Normal
from .actor_critic import get_activation
from .vel_estimator import VelEstimator


class VelActorCritic(nn.Module):
    """Actor-Critic with velocity estimator."""
    is_recurrent = False
    
    def __init__(self,
                 num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 num_one_step_obs=None,
                 actor_hidden_dims=[512, 256, 128],
                 critic_hidden_dims=[512, 256, 128],
                 activation='elu',
                 init_noise_std=1.0,
                 **kwargs):
        if kwargs:
            print("VelActorCritic.__init__ got unexpected arguments: " + str([key for key in kwargs.keys()]))
        super(VelActorCritic, self).__init__()

        activation = get_activation(activation)

        # If num_one_step_obs not provided, try to infer from frame_stack in kwargs
        if num_one_step_obs is None:
            frame_stack = kwargs.get('frame_stack', 10)
            num_one_step_obs = num_actor_obs // frame_stack
            print(f"VelActorCritic: Inferred num_one_step_obs={num_one_step_obs} from num_actor_obs={num_actor_obs} and frame_stack={frame_stack}")
        
        self.history_size = int(num_actor_obs / num_one_step_obs)
        self.num_actor_obs = num_actor_obs
        self.num_actions = num_actions
        self.num_one_step_obs = num_one_step_obs

        mlp_input_dim_a = num_one_step_obs + 3  # obs + vel
        mlp_input_dim_c = num_critic_obs

        # Estimator
        self.estimator = VelEstimator(
            temporal_steps=self.history_size,
            num_one_step_obs=num_one_step_obs
        )

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f'Velocity Estimator: {self.estimator.encoder}')

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs_history):
        with torch.no_grad():
            vel = self.estimator(obs_history)
        actor_input = torch.cat((obs_history[:, :self.num_one_step_obs], vel), dim=-1)
        mean = self.actor(actor_input)
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, obs_history=None, **kwargs):
        self.update_distribution(obs_history)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_history):
        vel = self.estimator(obs_history)
        actions_mean = self.actor(torch.cat((obs_history[:, :self.num_one_step_obs], vel), dim=-1))
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
