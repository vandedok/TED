import numpy as np
import torch
from TED.train import Workspace
import TED.utils as utils 

class ExternalAgentWorkspace(Workspace):

    def __init__(self, cfg, agent):
        super().__init__(cfg)

        action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        action_shape = self.env.action_space.shape
        obs_shape = self.env.observation_space.shape
        self.agent = agent(obs_shape, action_shape, action_range, cfg)

    def numpy_to_torch(self, arr):
        t = torch.tensor(arr).unsqueeze(0)
        if len(t.shape) == 1:
            t = t.unsqueeze(0)
        return t.to(self.device)
    
    def evaluate(self):
        average_episode_reward = 0

        dyn_lat_diff_losses = []
        dyn_rew_losses = []
        for episode in range(self.cfg.num_eval_episodes):

            obs = self.env.reset()

            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            episode_step = 0

            prev_obs = obs

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
 

                obs, reward, done, info = self.env.step(action)

                self.video_recorder.record(self.env)
                episode_reward += reward
                episode_step += 1

                with utils.eval_mode(self.agent):
                    # print("REW shape", reward, type(reward), reward.shape, self.numpy_to_torch(reward).shape, torch.tensor(reward).shape)
                    dyn_lat_diff_loss, dyn_rew_loss = self.agent.get_diff_lat_and_reward_losses(
                        self.numpy_to_torch(prev_obs), 
                        self.numpy_to_torch(obs), 
                        self.numpy_to_torch(action), 
                        self.numpy_to_torch(reward), 
                        scale_diff=False
                        )
                    dyn_lat_diff_losses.append(dyn_lat_diff_loss.item())
                    dyn_rew_losses.append(dyn_rew_loss.item())

                prev_obs = obs

            average_episode_reward += episode_reward
            self.video_recorder.save(f'{self.step}.mp4')

        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward, self.step)
        self.logger.log('eval/mean_lat_dif_loss', np.mean(dyn_lat_diff_losses), self.step)
        self.logger.log('eval/mean_rew_loss', np.mean(dyn_rew_losses), self.step)
        self.logger.dump(self.step)