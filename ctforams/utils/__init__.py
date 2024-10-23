from ctforams.utils.instantiators import instantiate_callbacks, instantiate_loggers
from ctforams.utils.logging_utils import log_hyperparameters
from ctforams.utils.pylogger import RankedLogger
from ctforams.utils.rich_utils import enforce_tags, print_config_tree
from ctforams.utils.utils import extras, get_metric_value, task_wrapper

from lightning.pytorch.plugins.environments import SLURMEnvironment


class DisabledSLURMEnvironment(SLURMEnvironment):
    def detect() -> bool:
        return False

    @staticmethod
    def _validate_srun_used() -> None:
        return

    @staticmethod
    def _validate_srun_variables() -> None:
        return
