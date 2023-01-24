from TED.algorithms.sac import SAC
from TED.algorithms.rad import RAD
from TED.algorithms.drq import DrQ
from TED.algorithms.svea import SVEA

algorithm = {
	'sac': SAC,
	'rad': RAD,
	'drq': DrQ,
	'svea': SVEA
}

def make_agent(obs_shape, action_shape, action_range, cfg):
	return algorithm[cfg.algorithm](obs_shape, action_shape, action_range, cfg)