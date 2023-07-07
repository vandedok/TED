from TED.train import Workspace


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