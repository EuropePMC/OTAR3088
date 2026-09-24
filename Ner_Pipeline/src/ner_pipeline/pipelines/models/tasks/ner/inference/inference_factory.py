
from dataclasses import dataclass

from ....shared.factory import BaseResolvedConfig
from ....shared.trainer_base import HFInferenceOrchestratorConfig



@dataclass
class NERInferenceResolvedConfig(BaseResolvedConfig):
    task_type: str = "ner_inference"
    


@dataclass(frozen=True)
class NERInferenceOrchestratorConfig(HFInferenceOrchestratorConfig):
    pass
