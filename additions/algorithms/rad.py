import TED.utils as utils
from .sac import DynAwareSAC

class DynAwareRAD(DynAwareSAC):
    def __init__(self, obs_shape, action_shape, action_range, cfg):
        super().__init__(obs_shape, action_shape, action_range, cfg)

    def replay_buffer_sample(self, replay_buffer):
        return replay_buffer.sample_rad(self.batch_size)
