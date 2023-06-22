import os
import time
import numpy as np
import torch
from TED.train import Workspace, make_env, try_import
import TED.utils as utils 


class EpisodeSaver():
    
    def __init__(self, out_dir, max_episodes=None, compress=False):
        self.set_out_dir(out_dir)
        self.episode_id = -2
        self.reset()
        if compress:
            self.saiving_f = np.savez_compressed
        else:
            self.saiving_f = np.savez

        self.max_episodes = max_episodes
        self.oldest_episode = 0
        self.saved_episodes = 0
        self.file_ext = "npz"

    def set_out_dir(self, out_dir):
        self.out_dir = out_dir
                    
    def reset(self):
        self.episode_id += 1
        self.observations = []
        self.actions = []
        self.rewards = []
        self.latents = []
        
    def add(self, obs, act, rew, lat):
        self.observations.append(obs)
        self.actions.append(act)
        self.rewards.append(rew)
        self.latents.append(lat)
        
    def save(self):
        observations = np.stack(self.observations)
        self.saved_episodes += 1
        actions = np.stack(self.actions)
        rewards = np.array(self.rewards)
        latents = np.stack(self.latents)
        self.saiving_f(
            os.path.join(self.out_dir, str(self.episode_id)), 
            obs = observations, 
            acts=actions,
            rews=rewards,
            lats = latents,
        )

        if not self.max_episodes is None and self.saved_episodes > self.max_episodes:
            os.remove(os.path.join(self.out_dir, str(self.oldest_episode)) + "." + self.file_ext)
            self.oldest_episode += 1
    
def ghost_pbar(x):
    return x
    
class EpisodesSavingWorkspace(Workspace):
    
    def __init__(self, cfg):
        super().__init__(cfg)

        self.episodes_dir = os.path.join(self.work_dir, "episodes")
        self.episodes_dir_before = os.path.join(self.episodes_dir , "before")
        self.episodes_dir_after = os.path.join(self.episodes_dir , "after")
        for dir_path in [self.episodes_dir, self.episodes_dir_before, self.episodes_dir_after]:
            os.makedirs(dir_path)           
        self.episode_saver_before = EpisodeSaver(
            self.episodes_dir_before, 
            compress=cfg.compress_saved_episodes,
            max_episodes = cfg.max_episodes_to_save,
            )
        self.episode_saver_after = EpisodeSaver(
            self.episodes_dir_after, 
            compress=cfg.compress_saved_episodes,
            max_episodes = cfg.max_episodes_to_save,
            )
        self.episodes_saver = self.episode_saver_before

    def is_train_env(self):
        if self.step < self.cfg.num_train_steps:
            return True
        else:
            return False
        
    def evaluate(self, pbar=ghost_pbar):
        average_episode_reward = 0

        for episode in pbar(range(self.cfg.num_eval_episodes)):

            obs = self.env.reset()

            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            episode_step = 0

            # reset episode_saver after the episdoe is finished
            
            if self.is_train_env():
                self.episode_saver = self.episode_saver_before
            else:
                self.episode_saver = self.episode_saver_after

            self.episode_saver.reset()

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)

                # collecting the latents for episode_saver
            
                obs, reward, done, info = self.env.step(action)
                # we are dealing with a batch of size [0]
                lats = self.get_latents(obs)[0]
                # adding s-a-r-l tuple to episode_saver
                self.episode_saver.add(obs, action, reward, lats)

                self.video_recorder.record(self.env)
                episode_reward += reward
                episode_step += 1

            # save episode_saver after the episdoe is finished
            self.episode_saver.save()

            average_episode_reward += episode_reward
            self.video_recorder.save(f'{self.step}.mp4')

        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward, self.step)
        self.logger.dump(self.step)
        
    def get_latents(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            output_logits = self.agent.actor.encoder.output_logits
            self.agent.actor.encoder.output_logits = True
            lats = self.agent.actor.encoder(
                obs,
                detach_encoder_conv=True,
                detach_encoder_head=True
            )
            self.agent.actor.encoder.output_logits = output_logits
        return lats.cpu().numpy()
 
    def run(self):
        episode, episode_reward, episode_step, done = 0, 0, 1, True
        start_time = time.time()

        total_num_steps = self.cfg.num_train_steps + self.cfg.num_test_steps

        while self.step <= total_num_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step % self.cfg.eval_freq == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward, self.step)

                obs = self.env.reset()

                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)
                    
            # # collecting the latents for episode_saver
            # lats = self.get_latents(obs)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                for _ in range(self.cfg.num_train_iters):
                    self.agent.update(self.replay_buffer, self.logger, self.step)

            if self.step > 0 and self.step % self.cfg.save_freq == 0:
                saveables = {
                    "actor": self.agent.actor.state_dict(),
                    "critic": self.agent.critic.state_dict(),
                    "critic_target": self.agent.critic_target.state_dict()
                }
                if self.cfg.ted:
                    saveables["ted_classifier"] = self.agent.ted_classifier.state_dict()
                save_at = os.path.join(self.save_dir, f"env_step{self.step * self.cfg.action_repeat}")
                os.makedirs(save_at, exist_ok=True)
                torch.save(saveables, os.path.join(save_at, "models.pt"))

            next_obs, reward, done, info = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done, done_no_max, episode)

            # # adding s-a-r-l tuple to episode_saver
            # self.episode_saver.add(obs, action, reward, lats)

            obs = next_obs
            episode_step += 1
            self.step += 1

            if self.step == self.cfg.num_train_steps:
                print("Switching to test env")
                self.env = self.test_env
                # Changing test_env_tracker for  episode_saver after the episdoe is finished
                self.episode_saver.set_out_dir(self.episodes_dir_after)

                done = True
    