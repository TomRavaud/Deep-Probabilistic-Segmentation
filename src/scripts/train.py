# Standard libraries
from typing import List
import sys

# Add the src directory to the system path
# (to avoid having to install project as a package)
sys.path.append("src/")

# Third-party libraries
import hydra
from omegaconf import DictConfig
from lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    Callback,
    seed_everything,
)
from lightning.pytorch.loggers import Logger
# import cv2

# Custom modules
from toolbox.utils.pylogger import RankedLogger
from toolbox.utils.instantiators import instantiate_callbacks, instantiate_loggers
from toolbox.utils.logging_utils import log_hyperparameters
from toolbox.utils.utils import get_metric_value


log = RankedLogger(__name__, rank_zero_only=True)


def train(cfg: DictConfig):
    """Perform training.

    Args:
        cfg (DictConfig): DictConfig object containing the configuration parameters.
    """
    # Set seed for random number generators in PyTorch, numpy and python.random
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    
    # datamodule.setup()
    
    # Get 1st data
    # batch = next(iter(datamodule.train_dataloader()))
    
    
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
    
    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))
    
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )
    
    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }
    
    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)


    if cfg.get("train"):
        log.info("Starting training...")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

        # model(batch)
    
    train_metrics = trainer.callback_metrics
    
    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")
    
    test_metrics = trainer.callback_metrics
    
    # Gather all metrics
    metric_dict = {**train_metrics, **test_metrics}
    
    return metric_dict, object_dict


@hydra.main(version_base="1.3",
            config_path="../../configs/",
            config_name="train.yaml")
def main(cfg: DictConfig):
    """Main entry point for training.

    Args:
        cfg (DictConfig): DictConfig object containing the configuration parameters.
    """
    # Train the model
    metric_dict, _ = train(cfg)
    
    # Safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    return metric_value


if __name__ == "__main__":
    main()
