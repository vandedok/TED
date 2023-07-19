import torch
from torch.nn import MSELoss
from torch import nn
from TED.algorithms.sac import SAC
import TED.utils as utils

class DynModel(nn.Module):
    
    def __init__(self, 
                 common_head_layers = [],
                 lats_diff_head_layers=[], 
                 reward_head_layers = [],  
                 batch_norm=True, 
                 lats_diff_scale =1.
        ):
        super().__init__()

        self.common_head = self.compose_mlp(common_head_layers, batch_norm)
        self.lats_diff_head = self.compose_mlp(lats_diff_head_layers, batch_norm)
        self.reward_head = self.compose_mlp(reward_head_layers, batch_norm)
        self.lats_diff_scale = lats_diff_scale

    def compose_mlp(self, dims, batch_norm):
        mlp = nn.Sequential()
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            mlp.add_module(f"Linear #{i}", nn.Linear(in_dim, out_dim))
            mlp.add_module(f"Activation #{i}", nn.ReLU())
            if batch_norm:
                mlp.add_module(f"Batch Norm #{i}",nn.BatchNorm1d(out_dim))
        return mlp

    def forward(self, lats, acts):
        x = torch.cat((acts, lats), dim=-1)
        common_rep = self.common_head(x)
        lats_diffs = self.lats_diff_head(common_rep)
        rewards = self.reward_head(common_rep)
        # lats_diffs = self.transform_lats_diffs(lats_diffs)
        return lats_diffs, rewards

class DynAwareSAC(SAC):

    def __init__(self, obs_shape, action_shape, action_range, cfg):
        super().__init__(obs_shape, action_shape, action_range, cfg)
        
        self.dyn_model = DynModel(
            common_head_layers= [cfg.feature_dim + action_shape[0]] + cfg.dyn_common_layers,
            lats_diff_head_layers = cfg.dyn_lats_diff_head_layers + [cfg.feature_dim],
            reward_head_layers = cfg.dyn_rew_head_layers + [1]
        ).to(self.device)
        self.dyn_params = list(self.dyn_model.parameters()) + list(self.critic.encoder.parameters())
        self.dyn_lat_loss = MSELoss()
        self.dyn_rew_loss = MSELoss()
        self.dyn_model_optimizer = torch.optim.Adam(self.dyn_params, lr=cfg.dyn_lr)
        self.dyn_lat_dif_scaling = cfg.dyn_lat_dif_scaling
        self.dyn_lat_coef = cfg.dyn_lat_coef
        self.dyn_rew_coef = cfg.dyn_rew_coef

    def update_dyn_model(self, obs, next_obs, action, reward, logger, step):

        obs_rep = self.critic.encoder(obs)
        with torch.no_grad():
            next_obs_rep = self.critic_target.encoder(next_obs)
            diff_gt = (next_obs_rep - torch.detach(obs_rep)) * self.dyn_lat_dif_scaling

        diff_pred, reward_pred = self.dyn_model(obs_rep, action)

        dyn_lat_diff_loss = self.dyn_lat_loss(diff_gt, diff_pred)
        dyn_rew_loss = self.dyn_rew_loss(reward, reward_pred)
        loss = self.dyn_lat_coef * dyn_lat_diff_loss + self.dyn_rew_coef * dyn_rew_loss
        loss.backward()
        self.dyn_model_optimizer.step()
   
        logger.log('train_dyn/dyn_lat_diff_loss', dyn_lat_diff_loss, step)
        logger.log('train_dyn/dyn_rew_loss', dyn_rew_loss, step)

    def update(self, replay_buffer, logger, step):
        ##### THis code is the same as in the original SAC, but I need the same samples from the replay buffer ###
        obs, action, reward, next_obs, not_done, same_episode_obs = replay_buffer.sample(self.batch_size)

        if self.ted:
            # Zero grad here as we will retain critic gradient
            self.ted_optimizer.zero_grad()

        logger.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, logger, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, logger, step)

        if self.ted:
            self.update_representation(obs, next_obs, same_episode_obs, replay_buffer, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic.Q1, self.critic_target.Q1, self.critic_tau)
            utils.soft_update_params(self.critic.Q2, self.critic_target.Q2, self.critic_tau)
            utils.soft_update_params(self.critic.encoder, self.critic_target.encoder, self.encoder_tau)

        #######

        self.update_dyn_model(obs, next_obs, action, reward, logger, step)

