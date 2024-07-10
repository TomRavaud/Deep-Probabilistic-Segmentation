# Standard libraries
import pathlib
import sys

# Add the src directory to the system path
# (to avoid having to install project as a package)
sys.path.append("src/")

# Third-party libraries
import hydra
from omegaconf import DictConfig
import torch
from tqdm import tqdm
import numpy as np

# # Custom modules
# from toolbox.utils.instantiators import instantiate_callbacks, instantiate_loggers
# from toolbox.utils.logging_utils import log_hyperparameters
# from toolbox.utils.utils import get_metric_value


def evaluate(cfg: DictConfig):
    """Perform evaluation.

    Args:
        cfg (DictConfig): DictConfig object containing the configuration parameters.
    """
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate the dataloader
    dataloader: torch.utils.data.DataLoader = hydra.utils.instantiate(cfg.data)
    
    # Instantiate the model
    model: torch.nn.Module = hydra.utils.instantiate(cfg.model)
    model.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Array to store the results
    results = np.empty((len(dataloader), 4), dtype="<U32")
    
    # Get logs directory
    runs_dir = pathlib.Path("logs/evaluate/runs")
    logs_dir = sorted(runs_dir.iterdir())[-1]
    
    # Go through the batches
    for i, batch in enumerate(tqdm(dataloader)):
        
        # Send the batch to the GPU if it is available
        batch = batch.to(device)
        
        # Perform the forward pass
        error, optimal_error = model(batch)
        
        # Store the results
        results[i] = np.array([[
            batch.object_labels[0],
            batch.scene_labels[0],
            error.item(),
            optimal_error.item() if optimal_error is not None else "N/A",
        ]])
        
        # Save the results
        if i % 50 == 0:
            np.save(logs_dir / "results.npy", results)
    
    np.save(logs_dir / "results.npy", results)
        
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
