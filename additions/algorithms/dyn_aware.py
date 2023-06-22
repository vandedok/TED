import torch
from torch.nn import MSELoss
from TED.algorithms.sac import SAC
from teddynamics.models import BasicDynModel

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

class DynAwareSAC(SAC):

    def __init__(self, obs_shape, action_shape, action_range, cfg):
        super.__init__(obs_shape, action_shape, action_range, cfg)
        self.dyn_lambda = cfg.dyn_lambda
        self.dyn_model = BasicDynModel(cfg.dyn_model_config)
        self.dyn_model_loss = MSELoss()

    def get_dyn_loss(self, obs, next_obs, rewards):
        obs_rep = self.actor.encoder(obs)
        next_obs_rep = self.actor.encoder(next_obs)
        next_obs_pred, _ = self.dyn_model(obs_rep,rewards)
        loss = self.dyn_model_loss(next_obs_pred, next_obs_rep) * self.dyn_lambda
        return loss


    def update_representation(self, obs, next_obs, same_episode_obs, replay_buffer, logger, step):
        num_samples = obs.shape[0]

        obs_rep = self.critic.encoder(obs)
        with torch.no_grad():
            next_obs_rep = self.critic_target.encoder(next_obs)

        

        # Stack the consecutive observations to make temporal samples
        non_iid_samples = torch.stack([obs_rep, next_obs_rep], dim=1)
        # All temporal samples are given a label of 1
        non_iid_labels = torch.ones((num_samples))

        # Create the non-temporal different episode samples
        rnd_idx = torch.randperm(num_samples)
        diff_ep_iid_samples = torch.stack([obs_rep, next_obs_rep[rnd_idx]], dim=1)
        # All non-temporal samples are given a label of 0
        diff_ep_iid_labels = torch.zeros((num_samples))

        # Create the non-temporal same episode samples
        with torch.no_grad():
            next_entry_rep = self.critic_target.encoder(same_episode_obs)
        same_ep_iid_samples = torch.stack([obs_rep, next_entry_rep], dim=1)
        same_ep_iid_labels = torch.zeros((num_samples))

        samples = torch.cat([non_iid_samples, diff_ep_iid_samples, same_ep_iid_samples])
        labels = torch.cat([non_iid_labels, diff_ep_iid_labels, same_ep_iid_labels]).to(self.device)

        r = self.ted_classifier(samples)

        dyn_loss = self.get_dyn_loss(obs_rep, next_obs_rep)

        ted_loss = self.ted_loss(r, labels) * self.ted_coef + dyn_loss

        logger.log('train_ted/loss', ted_loss, step)

        ted_loss.backward()
        self.ted_optimizer.step()
