# Standard libraries
import sys

# Add the src directory to the system path
# (to avoid having to install project as a package)
sys.path.append("src/")

# Third-party libraries
import hydra
from omegaconf import DictConfig
import torch

# # Custom modules
# from toolbox.utils.instantiators import instantiate_callbacks, instantiate_loggers
# from toolbox.utils.logging_utils import log_hyperparameters
# from toolbox.utils.utils import get_metric_value


def evaluate(cfg: DictConfig):
    """Perform evaluation.

    Args:
        cfg (DictConfig): DictConfig object containing the configuration parameters.
    """
    dataloader: torch.utils.data.DataLoader = hydra.utils.instantiate(cfg.data)
    
    print(dataloader)
    
    return


@hydra.main(version_base="1.3",
            config_path="../../configs/",
            config_name="evaluate.yaml")
def main(cfg: DictConfig):
    """Main entry point for evaluation.

    Args:
        cfg (DictConfig): DictConfig object containing the configuration parameters.
    """
    # Evaluate the model
    evaluate(cfg)
    
    return


if __name__ == "__main__":
    main()
