import torch
import timm
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import theseus as th
import copy
import os
import tqdm

# State Encoder (ho)
class StateEncoder(nn.Module):
    def __init__(self):
        super(StateEncoder, self).__init__()
        self.obs_encoder = timm.create_model(
            'resnet18.a1_in1k',
            pretrained=True,
            num_classes=0, 
        )
    def forward(self, nobs):
        return self.obs_encoder(nobs)

# Action Encoder (ha)
class ActionEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActionEncoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, action):
        return self.network(action)

# Fusing Encoder (hl)
class FusingEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FusingEncoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, hs, ha):
        h = torch.cat((hs, ha), dim=-1)
        h = self.network(h)
        mu, logvar = torch.split(h, int(0.5*h.shape[-1]), dim=-1)
        return mu, logvar

# Reward Decoder (r)
class RewardDecoder(nn.Module):
    def __init__(self, input_dim):
        super(RewardDecoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 1),
            # nn.Tanh()
        )

    def forward(self, z):
        return self.network(z)

# Dynamics Function (d)
class DynamicsFunction(nn.Module):
    def __init__(self, latent_dim_state, latent_dim_action, posterior_dim):
        super(DynamicsFunction, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim_state + latent_dim_action + posterior_dim, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, latent_dim_state + posterior_dim)
        )
        self.latent_dim_state = latent_dim_state

    def forward(self, zs, za, z):
        z_combined = torch.cat((zs, za, z), dim=-1)
        z = self.network(z_combined)
        return z[:, :self.latent_dim_state], z[:, self.latent_dim_state:]

# Decoder for Action Reconstruction
class ActionDecoder(nn.Module):
    def __init__(self, posterior_dim, latent_dim_state, action_dim):
        super(ActionDecoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(posterior_dim + latent_dim_state, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, z, zs):
        z_combined = torch.cat((z, zs), dim=-1)
        return self.network(z_combined)


# Loss Function (ELBO)
def compute_elbo_loss(predict_actions, na_target, mu, logvar):
    # Reconstruction loss (MSE)
    l2_loss = nn.MSELoss()(predict_actions, na_target)
    # KL divergence
    kl_loss = -10 * torch.mean(0.5 * torch.sum(1 + logvar - logvar.exp() - mu.pow(2), dim=1), dim=0)

    return l2_loss, kl_loss

class Normalizer(object):
    """Normalizes observations and actions to zero mean and unit variance."""
    def __init__(self,cfg):
        self.obs_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        self.obs_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
        self.action_mean = torch.zeros(cfg.a_dim).cuda()
        self.action_std = 10*torch.ones(cfg.a_dim).cuda()

    def normalize_obs(self, obs):
        return (obs - self.obs_mean) / (self.obs_std + 1e-8)

    def normalize_action(self, action):
        return (action - self.action_mean) / (self.action_std + 1e-8)

    def denormalize_obs(self, obs):
        return obs * self.obs_std + self.obs_mean

    def denormalize_action(self, action):
        return action * self.action_std + self.action_mean

import torch
import torch.nn as nn
import torch.nn.functional as F13
import theseus as th
# from cvae_utilities import *
import os
import wandb
import matplotlib.pyplot as plt
from copy import deepcopy

class CVAEWithTrajectoryOptimization(nn.Module):
    def __init__(self, cfg):
        super(CVAEWithTrajectoryOptimization, self).__init__()
        s_dim = cfg.s_dim
        a_dim = cfg.a_dim
        horizon = cfg.horizon
        z_dim = cfg.z_dim
        device = cfg.device
        self.state_encoder = StateEncoder().to(device)
        self.action_encoder = ActionEncoder(a_dim * horizon, a_dim * horizon).to(device)
        self.fusing_encoder = FusingEncoder(s_dim + a_dim * horizon, z_dim*2).to(device)
        self.reward_decoder = RewardDecoder(s_dim + a_dim * horizon + z_dim).to(device)
        self.z_dim = z_dim
        self.device = device
        self.cfg = cfg
        self.normalizer = Normalizer(cfg)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def plan_action(self,
                init_actions,
                z,
                batch_size,
                horizon,
                action_dim,
                damping,
                traj_opt_num,
                traj_opt_step,
                train_mode=True):
            actions = init_actions.detach().reshape(1, -1)
            init_actions = deepcopy(actions)
            actions = th.Vector(tensor=actions, name="actions")
            z = th.Variable(tensor=z, name="z")
            def value_cost_fn(optim_vars, aux_vars):
                temp_actions = optim_vars[0].tensor.view(batch_size, horizon * action_dim)
                temp_actions = torch.clamp(temp_actions, -1, 1)
                temp_z = aux_vars[0].tensor.view(batch_size, -1)
                x = torch.cat([temp_z, temp_actions], dim=-1)
                reward = self.reward_decoder(x)
                err = -reward.nan_to_num_(0) + 1000
                err = err.mean(dim=0).view(1, 1)
                return err

            cost_fn = th.AutoDiffCostFunction(
                [actions], value_cost_fn, 1, aux_vars=[z], name="value_cost_fn"
            )

            objective = th.Objective()
            objective.to(device=self.device)
            objective.add(cost_fn)

            optimizer = th.LevenbergMarquardt(
                objective, th.CholeskyDenseSolver,
                max_iterations=traj_opt_num,
                step_size=traj_opt_step,
            )

            theseus_layer = th.TheseusLayer(optimizer).to(device=self.device)

            theseus_inputs = {"actions": init_actions, "z": z.tensor}

            if train_mode:
                updated_inputs, info = theseus_layer.forward(
                    theseus_inputs,
                    optimizer_kwargs={
                        "track_best_solution": True,
                        "damping": damping,
                        "verbose": False,
                        "backward_mode": "truncated",
                        "backward_num_iterations": 5,
                    },
                )
            else:
                with torch.no_grad():
                    updated_inputs, info = theseus_layer.forward(
                        theseus_inputs,
                        optimizer_kwargs={
                            "track_best_solution": True,
                            "damping": damping,
                            "verbose": False,
                            "backward_mode": "truncated",
                            "backward_num_iterations": 5,
                        },
                    )
            updated_actions = updated_inputs["actions"].view(batch_size, horizon, action_dim)
            predict_actions = updated_actions#TODO: multi planing horizon
            return predict_actions, info

    
    def save_pretrained(self, save_directory):
        """Save model weights and config to directory"""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model state dict
        model_path = os.path.join(save_directory, "model.pt")
        torch.save({
            'state_encoder': self.state_encoder.state_dict(),
            'action_encoder': self.action_encoder.state_dict(),
            'fusing_encoder': self.fusing_encoder.state_dict(),
            'dynamics_function': self.dynamics_function.state_dict(),
            'action_decoder': self.action_decoder.state_dict(),
            'reward_decoder': self.reward_decoder.state_dict(),
            'cfg': self.cfg
        }, model_path)

    def load_pretrained(self, load_directory):
        """Load model weights and config from directory"""
        model_path = os.path.join(load_directory, "model.pt")
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found at {model_path}")
            
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load model weights
        self.state_encoder.load_state_dict(checkpoint['state_encoder'])
        self.action_encoder.load_state_dict(checkpoint['action_encoder'])
        self.fusing_encoder.load_state_dict(checkpoint['fusing_encoder'])
        self.dynamics_function.load_state_dict(checkpoint['dynamics_function'])
        self.action_decoder.load_state_dict(checkpoint['action_decoder'])
        self.reward_decoder.load_state_dict(checkpoint['reward_decoder'])
        
        # Load config
        self.cfg = checkpoint['cfg']

    def plan_with_theseus_update(self, obs, state, actions = None, train_mode=True):
        """
        Perform trajectory optimization using Theseus.
        Args:
            obs: Observation (raw state input).
            horizon: Planning horizon.
            gamma: Discount factor.
            model: The CVAE model containing dynamics and state encoders.
            cfg: Configuration object (for damping, step size, etc.).
            eval_mode: Whether to evaluate without gradients.
        """
        # Prepare initial observation
        batch_size = obs.shape[0]
        obs = deepcopy(obs[:,0,...]) # [bs, 3, 96, 96]
        nobs = self.normalizer.normalize_obs(obs)
        cur_actions = deepcopy(state) # [bs, 1, 2]
        # ncactions = self.normalizer.normalize_action(cur_actions)

        #TODO: can use diff of action
        if train_mode:
            actions = deepcopy(actions) 
            actions = (actions - torch.concat([cur_actions, actions[:,:-1,:]], dim=1)) #action diff
            na_target = self.normalizer.normalize_action(actions) # [bs, horizon, 2]
            nactions = na_target.reshape(batch_size, -1) # [bs, horizon * 2]
            hs = self.state_encoder(nobs) # [bs, s_dim]
            ha = self.action_encoder(nactions) # [bs, a_dim * horizon]
            mu, logvar = self.fusing_encoder(hs, ha) # [bs, z_dim]
            z = self.reparameterize(mu, logvar) # [bs, z_dim]
            
            z = torch.cat([hs, z], dim=-1) # [bs, s_dim + z_dim]
            z = z.reshape(1,-1)
            init_actions = nactions + torch.randn_like(nactions) * 0.05
        else:
            with torch.no_grad():
                obs_features = self.state_encoder(nobs)
                z = torch.randn(batch_size, self.z_dim).to(self.device)
                z = torch.cat([obs_features, z], dim=-1)
                z = z.reshape(1, -1)
                mu = torch.zeros(batch_size, self.z_dim).to(self.device)
                logvar = torch.zeros(batch_size, self.z_dim).to(self.device)
                na_target = torch.zeros(batch_size, self.cfg.horizon, self.cfg.a_dim).to(self.device)
                init_actions = torch.zeros(batch_size, self.cfg.horizon * self.cfg.a_dim).to(self.device)
                #TODO: multi planing horizon

        predict_actions, info = self.plan_action(
            init_actions,
            z,
            batch_size,
            self.cfg.horizon,
            self.cfg.a_dim,
            self.cfg.damping,
            self.cfg.traj_opt_num,
            self.cfg.traj_opt_step,
            train_mode=train_mode
        )
        l2_loss, kl_loss = compute_elbo_loss(
            predict_actions, na_target, mu, logvar
        )
        output_dict = {
            "best_actions": predict_actions,
            "l2_loss": l2_loss,
            "kl_loss": kl_loss,
            "info": info
        }

        return output_dict

def evaluate(policy, eval_dataloader):
    torch.cuda.empty_cache()  # Clear cache before evaluation
    policy.eval()
    total_loss = 0
    num_batches = 0
    device = policy.device
    with torch.no_grad():
        for batch in tqdm.tqdm(eval_dataloader, desc="Evaluating"):
            obs = batch['observation.image'].to(device, non_blocking=True)
            state = batch['observation.state'].to(device, non_blocking=True)
            action = batch['action'].to(device, non_blocking=True)
            output_dict = policy.plan_with_theseus_update(obs, state, train_mode=False)
            total_loss += output_dict["l2_loss"].item()
            num_batches += 1
            if num_batches >= 10: 
                break
            del output_dict  
            torch.cuda.empty_cache()
    policy.train()
    return total_loss / num_batches