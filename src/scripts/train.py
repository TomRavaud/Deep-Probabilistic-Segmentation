# Standard libraries
import sys

# Add the src directory to the system path
# (to avoid having to install project as a package)
sys.path.append("src/")

# Third-party libraries
import hydra
from omegaconf import DictConfig
from lightning import LightningDataModule


def train(cfg: DictConfig):
    """Perform training.

    Args:
        cfg (DictConfig): DictConfig object containing the configuration parameters.
    """
    # # Set seed for random number generators in pytorch, numpy and python.random
    # if cfg.get("seed"):
    #     L.seed_everything(cfg.seed, workers=True)

    # log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    
    datamodule.setup()
    
    # Get 1st data
    batch = next(iter(datamodule.train_dataloader()))
    
    print(batch.rgbs.shape,
          len(batch.object_datas),
          batch.bboxes.shape,
          batch.TCO.shape,
          batch.K.shape,
          batch.depths)

    

@hydra.main(version_base="1.3",
            config_path="../../configs/",
            config_name="train.yaml")
def main(cfg: DictConfig):
    """Main entry point for training.

    Args:
        cfg (DictConfig): DictConfig object containing the configuration parameters.
    """
    train(cfg)


if __name__ == "__main__":
    main()
