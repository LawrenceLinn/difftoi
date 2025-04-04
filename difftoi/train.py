import torch, os, wandb
import tqdm, numpy as np
from torch import nn
from torch.utils.data import DataLoader
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
import gymnasium as gym
import gym_pusht
import collections
from cvae_utilities import *

# torch.cuda.set_per_process_memory_fraction(0.9)  
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class Config:
    def __init__(self, config_dict):
        for k, v in config_dict.items():
            setattr(self, k, v)

config_dict = {
    "horizon": 7, "s_dim": 512, "a_dim": 2, "z_dim": 64,
    "traj_opt_num": 50, "traj_opt_step": 1e-2, "damping": 1e-3,
    "batch_size": 16, "device": torch.device("cuda"),
    "beta": 1.0, "learning_rate": 1e-4, "num_epochs": 1000,
    "gradient_accumulation_steps": 4, "log_freq": 1,
    "eval_freq": 50, "output_directory": "./output",
    "online_eval_episodes": 5,
}

config = Config(config_dict)

wandb.init(
    project="difftop-training",
    config=config_dict,
    # resume=True
)

device = torch.device("cuda")
policy = CVAEWithTrajectoryOptimization(config)

# Load previous checkpoint if exists
if os.path.exists(f"{wandb.config.output_directory}/best_model"):
    print("Loading previous checkpoint...")
    policy.load_pretrained(f"{wandb.config.output_directory}/best_model")

policy.train()
policy.to(device)

optimizer = torch.optim.Adam(policy.parameters(), lr=wandb.config.learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                        T_max=15000,
                                                        eta_min=1e-6)

# Set up the dataset
delta_timestamps = {
    "observation.image": [0.0],
    "observation.state": [0.0],
    "action": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
}

dataset = LeRobotDataset("lerobot/pusht_image", delta_timestamps=delta_timestamps)

episode_indices_to_use = np.arange(dataset.num_episodes)
test_episode_indices = np.random.choice(episode_indices_to_use, 10, replace=False)
train_episode_indices = np.setdiff1d(episode_indices_to_use, test_episode_indices)

train_sampler = EpisodeAwareSampler(dataset.episode_data_index, episode_indices_to_use=train_episode_indices, shuffle=True)
test_sampler = EpisodeAwareSampler(dataset.episode_data_index, episode_indices_to_use=test_episode_indices, shuffle=True)

train_dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=4,  # Reduced number of workers
    batch_size=wandb.config.batch_size,
    sampler=train_sampler,
    pin_memory=True,
    drop_last=True,
)

eval_dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=4,  # Reduced number of workers
    batch_size=wandb.config.batch_size,
    sampler=test_sampler,
    pin_memory=True,
    drop_last=True,
)

global_step = 0
for epoch in range(wandb.config.num_epochs):
    print(f"Epoch {epoch+1}/{wandb.config.num_epochs}")
    epoch_loss = 0
    num_batches = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
        torch.cuda.empty_cache()  # Clear cache before each batch
        # batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        obs = batch['observation.image'].to(device, non_blocking=True)
        state = batch['observation.state'].to(device, non_blocking=True)
        action = batch['action'].to(device, non_blocking=True)
        output_dict = policy.plan_with_theseus_update(obs, state, action, train_mode=True)
        bc_loss = output_dict["l2_loss"] / wandb.config.gradient_accumulation_steps
        kl_loss = output_dict["kl_loss"] / wandb.config.gradient_accumulation_steps
        loss = bc_loss + kl_loss
        loss.backward()
        epoch_loss += loss.item() * wandb.config.gradient_accumulation_steps
        num_batches += 1

        # Log the min and max of the best actions
        best_actions = output_dict['best_actions'][0].reshape(7, 2)
        wandb.log({
            "train/min_best_action": best_actions.min(),
            "train/max_best_action": best_actions.max()
        })

        # Free memory
        del output_dict
        torch.cuda.empty_cache()

        if global_step % wandb.config.gradient_accumulation_steps == 0 and global_step > 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
        if global_step % wandb.config.log_freq == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.3f}, LR: {current_lr:.6f}")
            wandb.log({
                "train/batch_loss": loss.item(),
                "train/bc_loss":bc_loss.item(),
                "train/learning_rate": current_lr,
                "train/batch": batch_idx + 1,
                "train/global_step": global_step
            })

        # Evaluate every eval_freq steps
        if global_step % wandb.config.eval_freq == 0:
            # Offline evaluation
            eval_loss = evaluate(policy, eval_dataloader)
            print(f"Step {global_step} Eval Loss: {eval_loss:.3f}")
            
            # Online evaluation
            # mean_reward, mean_length = online_evaluate(policy, num_episodes=wandb.config.online_eval_episodes)
            # print(f"Online Eval - Mean Reward: {mean_reward:.3f}, Mean Episode Length: {mean_length:.1f}")
            
            wandb.log({
                "eval/loss": eval_loss,
                # "eval/online_reward": mean_reward,
                # "eval/episode_length": mean_length,
                "eval/global_step": global_step
            })

            #     # Save best model based on both metrics
            #     if eval_loss < best_eval_loss or mean_reward > best_online_reward:
            #         best_eval_loss = min(eval_loss, best_eval_loss)
            #         best_online_reward = max(mean_reward, best_online_reward)
            #         policy.save_pretrained(f"{wandb.config.output_directory}/best_model")
            #         wandb.log({
            #             "eval/best_loss": best_eval_loss,
            #             "eval/best_online_reward": best_online_reward,
            #             "eval/best_step": global_step
            #         })

        global_step += 1
        print(f"Global Step: {global_step}")

    # End of epoch
    avg_epoch_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss:.3f}")
    wandb.log({
        "train/epoch_loss": avg_epoch_loss,
        "train/epoch": epoch + 1
    })