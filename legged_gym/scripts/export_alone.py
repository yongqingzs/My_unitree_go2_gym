#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
import sys
import argparse
# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
# Import torch first to avoid isaacgym import issues
import torch
import copy

from rsl_rl.modules import ActorCritic


class PolicyExporter(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)

    def forward(self, obs):
        return self.actor(obs)


def infer_model_dims(checkpoint):
    if 'model_state_dict' in checkpoint:
        model_params = checkpoint['model_state_dict']
    else:
        model_params = checkpoint
    
    # Actor hidden dims
    actor_0_weight = model_params['actor.0.weight']
    hidden_dims_actor = [
        actor_0_weight.shape[0],
        model_params['actor.2.weight'].shape[0],
        model_params['actor.4.weight'].shape[0]
    ]
    
    # Critic hidden dims
    critic_0_weight = model_params['critic.0.weight']
    hidden_dims_critic = [
        critic_0_weight.shape[0],
        model_params['critic.2.weight'].shape[0],
        model_params['critic.4.weight'].shape[0]
    ]
    
    num_actor_obs = actor_0_weight.shape[1]
    num_critic_obs = critic_0_weight.shape[1]
    num_actions = model_params['actor.6.weight'].shape[0]
    
    return num_actor_obs, num_critic_obs, num_actions, hidden_dims_actor, hidden_dims_critic


def export_policy(checkpoint_path, output_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
    else:
        model_state = checkpoint
    
    num_actor_obs, num_critic_obs, num_actions, hidden_dims_actor, hidden_dims_critic = infer_model_dims(checkpoint)
    
    print(f"Model dimensions: obs={num_actor_obs}, critic_obs={num_critic_obs}, actions={num_actions}")
    print(f"Actor hidden dims: {hidden_dims_actor}")
    print(f"Critic hidden dims: {hidden_dims_critic}")
    
    actor_critic = ActorCritic(num_actor_obs, num_critic_obs, num_actions, actor_hidden_dims=hidden_dims_actor, critic_hidden_dims=hidden_dims_critic)
    model_dict = actor_critic.state_dict()
    pretrained_dict = {k: v for k, v in model_state.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    actor_critic.load_state_dict(model_dict)
    actor_critic.eval()
    
    exporter = PolicyExporter(actor_critic)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    exporter.to('cpu')
    traced_script_module = torch.jit.script(exporter)
    traced_script_module.save(output_path)
    
    file_size = os.path.getsize(output_path) / 1024
    print(f"✓ Policy exported to: {output_path} ({file_size:.2f} KB)")
    
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint .pt file')
    parser.add_argument('--output', type=str, default='policy.pt', help='Output path')
    args = parser.parse_args()
    
    export_policy(args.checkpoint, args.output)