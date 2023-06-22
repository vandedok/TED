import os
from TED.train import Workspace
from .episode_saving import EpisodesSavingWorkspace
from teddynamics.models import BasicDynModel
from teddynamics.trainers import DynTrainer
from teddynamics.utils import load_yml

class DynAwareWorkspace(EpisodesSavingWorkspace):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.dyn_model_config = load_yml(cfg.dyn_model_config)
        self.dyn_model = BasicDynModel(
            lats_diff_head_layers=self.dyn_model_config ["model"]["lats_diff_head_layers"], 
            lats_diff_scale=self.dyn_model_config ["model"]["lats_diff_scale_coef"])
        self.episodes_dir = os.path.join(self.work_dir, "episodes")
        self.dyn_model_logs_dir = os.path.join(self.work_dir, "dyn_model")
        


    def evaluate(self):
        super().evaluate()

        if self.is_train_env():
            episodes_dir = self.episodes_dir_before 
        else:
            episodes_dir = self.episodes_dir_after
 

        dyn_trainer = DynTrainer(
            config = self.dyn_model_config, 
            episodes_dir = episodes_dir,
            experiment_dir = os.path.join(self.dyn_model_logs_dir),
            model=self.dyn_model,
            overwrite=True,
            )
        
        dyn_trainer.train()

