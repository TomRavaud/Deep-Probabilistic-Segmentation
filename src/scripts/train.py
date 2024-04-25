# Standard libraries
# from typing import List
import sys

# Add the src directory to the system path
# (to avoid having to install project as a package)
sys.path.append("src/")

# Third-party libraries
import hydra
from omegaconf import DictConfig
from lightning import LightningDataModule, LightningModule
# from lightning.pytorch.loggers import Logger

# import cv2

# Custom modules
from toolbox.utils.pylogger import RankedLogger


log = RankedLogger(__name__, rank_zero_only=True)


def train(cfg: DictConfig):
    """Perform training.

    Args:
        cfg (DictConfig): DictConfig object containing the configuration parameters.
    """
    # # Set seed for random number generators in pytorch, numpy and python.random
    # if cfg.get("seed"):
    #     L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    
    datamodule.setup()
    
    # Get 1st data
    batch = next(iter(datamodule.train_dataloader()))
    
    # print(batch.rgbs.shape,
    #       len(batch.object_datas),
    #       batch.bboxes.shape,
    #       batch.TCO.shape,
    #       batch.K.shape,
    #       batch.depths,
    #       "\n")
    
    # # Visualize the data
    # for i in range(cfg.data.dataloader_cfg.batch_size):
        
    #     print(batch.object_datas[i].label)
        
    #     # Get the image
    #     img = batch.rgbs[i].permute(1, 2, 0).numpy()
        
    #     # Image to BGR
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
    #     # Draw the bounding box
    #     bbox = batch.bboxes[i].numpy()
    #     bbox = bbox.astype(int)
    #     cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
    #     cv2.imshow("Image", img)
    #     cv2.waitKey(0)
    
    
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model) 

    if cfg.get("train"):
        log.info("Starting training...")
        
        model(batch)

    

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
