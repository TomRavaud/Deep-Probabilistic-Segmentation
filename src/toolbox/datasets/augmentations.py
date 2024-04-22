from __future__ import annotations

# Standard librairies
import dataclasses
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod

# Third party librairies
import cv2
import numpy as np
import PIL
import torch
from torchvision.datasets import VOCSegmentation
from torch.utils.data import Dataset

# Custom modules
from toolbox.datasets.scene_dataset import Resolution, SceneObservation
from toolbox.geometry.camera_geometry import get_K_crop_resize


class SceneObservationTransform(ABC):
    """
    Abstract base class for scene observation transformations.
    """
    def __init__(self, p: float = 1.0) -> None:
        """Constructor.

        Args:
            p (float, optional): Probability of applying the transformation.
                Defaults to 1.0.
        """
        self._p = p
    
    def __call__(self, obs: SceneObservation) -> SceneObservation:
        """Apply or not the transformation to the observation given the
        probability `p`.

        Args:
            obs (SceneObservation): Scene observation.

        Returns:
            SceneObservation: Eventually transformed scene observation.
        """
        if random.random() <= self._p:
            return self._transform(obs)
        else:
            return obs
    
    @abstractmethod
    def _transform(self, obs: SceneObservation) -> SceneObservation:
        """Define the transformation to apply to the observation.

        Args:
            obs (SceneObservation): Scene observation.

        Returns:
            SceneObservation: Transformed scene observation.
        """
        pass


class ComposeSceneObservationTransform(SceneObservationTransform):
    """
    Composing multiple scene observation transformations.
    """
    def __init__(
        self,
        transforms: List[SceneObservationTransform],
        p: float = 1.0,
    ) -> None:
        """Constructor.

        Args:
            transforms (List[SceneObservationTransform]): List of scene observation
                transformations to apply.
            p (float, optional): Probability of applying the transformation.
                Defaults to 1.0.
        """
        super().__init__(p)
        
        self._transforms = transforms
    
    def _transform(self, obs: SceneObservation) -> SceneObservation:
        """Apply the list of transformations to the observation.

        Args:
            obs (SceneObservation): Scene observation.

        Raises:
            ValueError: If the transforms are not a list of SceneObservationTransform.

        Returns:
            SceneObservation: Transformed scene observation.
        """
        if not isinstance(self._transforms, list):
            raise ValueError(
                "The transforms must be a list of SceneObservationTransform."
                )
        
        for transform in self._transforms:
            obs = transform(obs)
        
        return obs


#-------------------#
# RGB augmentations #
#-------------------#

class PillowRGBTransform(SceneObservationTransform):
    """
    Base class for PIL RGB transformations.
    """
    def __init__(
        self,
        pillow_fn: PIL.ImageEnhance._Enhance,
        factor_interval: Tuple[float, float],
        p: float = 1.0,
    ) -> None:
        """Constructor.

        Args:
            pillow_fn (PIL.ImageEnhance._Enhance): PIL image enhancement function.
            factor_interval (Tuple[float, float]): Interval of enhancement factor.
            p (float, optional): Probability of applying the transformation.
                Defaults to 1.0.
        """
        super().__init__(p)
        
        self._pillow_fn = pillow_fn
        self._factor_interval = factor_interval

    def _transform(self, obs: SceneObservation) -> SceneObservation:
        """Apply the PIL transformation to the RGB observation.

        Args:
            obs (SceneObservation): Scene observation.

        Returns:
            SceneObservation: Transformed scene observation.
        """
        # Get the PIL image from the RGB observation
        rgb_pil = PIL.Image.fromarray(obs.rgb)
        
        # Apply the PIL transformation
        rgb_pil = self._pillow_fn(rgb_pil).enhance(
            factor=random.uniform(*self._factor_interval),
        )
        
        # Replace the RGB observation with the transformed one
        obs = dataclasses.replace(obs, rgb=np.array(rgb_pil))
        
        return obs

class PillowSharpness(PillowRGBTransform):
    """
    PIL RGB sharpness transformation.
    """
    def __init__(
        self,
        factor_interval: Tuple[float, float] = (0.0, 50.0),
        p: float = 1.0,
    ) -> None:
        """Constructor.

        Args:
            factor_interval (Tuple[float, float], optional): Interval of enhancement
                factor. Defaults to (0.0, 50.0).
            p (float, optional): Probability of applying the transformation.
                Defaults to 1.0.
        """
        super().__init__(
            pillow_fn=PIL.ImageEnhance.Sharpness,
            factor_interval=factor_interval,
            p=p,
        )

class PillowContrast(PillowRGBTransform):
    """
    PIL RGB contrast transformation.
    """
    def __init__(
        self,
        factor_interval: Tuple[float, float] = (0.2, 50.0),
        p: float = 1.0,
    ) -> None:
        """Constructor.

        Args:
            factor_interval (Tuple[float, float], optional): Interval of enhancement
                factor. Defaults to (0.2, 50.0).
            p (float, optional): Probability of applying the transformation.
                Defaults to 1.0.
        """
        super().__init__(
            pillow_fn=PIL.ImageEnhance.Contrast,
            factor_interval=factor_interval,
            p=p,
        )

class PillowBrightness(PillowRGBTransform):
    """
    PIL RGB brightness transformation.
    """
    def __init__(
        self,
        factor_interval: Tuple[float, float] = (0.1, 6.0),
        p: float = 1.0,
    ) -> None:
        """Constructor.

        Args:
            factor_interval (Tuple[float, float], optional): Interval of enhancement
                factor. Defaults to (0.1, 6.0).
            p (float, optional): Probability of applying the transformation.
                Defaults to 1.0.
        """
        super().__init__(
            pillow_fn=PIL.ImageEnhance.Brightness,
            factor_interval=factor_interval,
            p=p,
        )

class PillowColor(PillowRGBTransform):
    """
    PIL RGB color transformation.
    """
    def __init__(
        self,
        factor_interval: Tuple[float, float] = (0, 20.0),
        p: float = 1.0,
    ) -> None:
        """Constructor.

        Args:
            factor_interval (Tuple[float, float], optional): Interval of enhancement
                factor. Defaults to (0, 20.0).
            p (float, optional): Probability of applying the transformation.
                Defaults to 1.0.
        """
        super().__init__(
            pillow_fn=PIL.ImageEnhance.Color,
            factor_interval=factor_interval,
            p=p,
        )

class PillowBlur(SceneObservationTransform):
    """
    Blur the RGB observation.
    """
    def __init__(
        self,
        factor_interval: Tuple[int, int] = (1, 3),
        p: float = 1.0,
    ) -> None:
        """Constructor.

        Args:
            factor_interval (Tuple[int, int], optional): Interval of blur factor.
                Defaults to (1, 3).
            p (float, optional): Probability of applying the transformation.
                Defaults to 1.0.
        """
        super().__init__(p)
        
        self._factor_interval = factor_interval

    def _transform(self, obs: SceneObservation) -> SceneObservation:
        """Apply a Gaussian blur to the RGB observation.

        Args:
            obs (SceneObservation): Scene observation.

        Raises:
            ValueError: If the RGB observation is None.

        Returns:
            SceneObservation: Transformed scene observation.
        """
        if obs.rgb is None:
            raise ValueError("The RGB observation is None.")
        
        # Get the PIL image from the RGB observation
        rgb_pil = PIL.Image.fromarray(obs.rgb)
        
        # Set a random blur factor
        k = random.randint(*self.factor_interval)
        
        # Apply a Gaussian blur to the image
        rgb_pil = rgb_pil.filter(PIL.ImageFilter.GaussianBlur(k))
        
        # Replace the RGB observation with the transformed one
        obs = dataclasses.replace(obs, rgb=np.array(rgb_pil))
        
        return obs


# --------------------#
# Depth augmentations #
# --------------------#

class DepthTransform(SceneObservationTransform, ABC):
    """
    Base class for depth transformations.
    """
    def _transform(self, obs: SceneObservation) -> SceneObservation:
        """Apply the depth transformation to the observation.

        Args:
            obs (SceneObservation): Scene observation.

        Raises:
            ValueError: If the depth observation is None.

        Returns:
            SceneObservation: Transformed scene observation.
        """
        if obs.depth is None:
            raise ValueError("The depth observation is None.")
        
        depth = self._transform_depth(obs.depth)
        
        # Replace the depth observation with the transformed one
        obs = dataclasses.replace(obs, depth=depth)
        
        return obs
    
    @abstractmethod
    def _transform_depth(self, depth: np.ndarray) -> np.ndarray:
        """Define the transformation to apply to the depth observation.

        Args:
            depth (np.ndarray): Depth observation.

        Returns:
            np.ndarray: Transformed depth observation.
        """
        pass

class DepthGaussianNoiseTransform(DepthTransform):
    """
    Add random Gaussian noise to the depth image.
    """
    def __init__(self, std_dev: float = 0.02, p: float = 1.0) -> None:
        """Constructor.

        Args:
            std_dev (float, optional): Standard deviation of the Gaussian noise.
                Defaults to 0.02.
            p (float, optional): Probability of applying the transformation.
                Defaults to 1.0.
        """
        super().__init__(p)
        
        self._std_dev = std_dev

    def _transform_depth(self, depth: np.ndarray) -> np.ndarray:
        """Add random Gaussian noise to the depth image.

        Args:
            depth (np.ndarray): Depth image.

        Returns:
            np.ndarray: Depth image with added noise.
        """
        depth = np.copy(depth)
        
        # Get a random Gaussian noise map
        noise = np.random.normal(scale=self._std_dev, size=depth.shape)
        
        # Add the noise to the depth image where the depth is greater than 0
        depth[depth > 0] += noise[depth > 0]
        
        # Clip the depth values between 0 and the maximum float32 value
        # (depth values are always positive)
        depth = np.clip(depth, 0, np.finfo(np.float32).max)
        
        return depth

class DepthCorrelatedGaussianNoiseTransform(DepthTransform):
    """
    Add random Gaussian noise to the depth image.
    """
    def __init__(
        self,
        std_dev: float = 0.01,
        gp_rescale_factor_min: float = 15.0,
        gp_rescale_factor_max: float = 40.0,
        p: float = 1.0,
    ) -> None:
        """Constructor.

        Args:
            std_dev (float, optional): Standard deviation of the Gaussian noise.
                Defaults to 0.01.
            gp_rescale_factor_min (float, optional): Minimum Gaussian noise map rescale
                factor. Defaults to 15.0.
            gp_rescale_factor_max (float, optional): Maximum Gaussian noise map rescale
                factor. Defaults to 40.0.
            p (float, optional): Probability of applying the transformation.
                Defaults to 1.0.
        """
        super().__init__(p)
        
        self._std_dev = std_dev
        self._gp_rescale_factor_min = gp_rescale_factor_min
        self._gp_rescale_factor_max = gp_rescale_factor_max

    def _transform_depth(self, depth: np.ndarray) -> np.ndarray:
        """Add random Gaussian noise to the depth image.

        Args:
            depth (np.ndarray): Depth image.

        Returns:
            np.ndarray: Depth image with added noise.
        """
        
        H, W = depth.shape
        depth = np.copy(depth)
        
        # Set a random rescale factor
        rescale_factor = np.random.uniform(
            low=self._gp_rescale_factor_min,
            high=self._gp_rescale_factor_max,
        )

        # Get the rescaled dimensions
        small_H, small_W = (np.array([H, W]) / rescale_factor).astype(int)
        
        # Sample a random Gaussian noise map with the rescaled dimensions
        additive_noise = np.random.normal(
            loc=0.0,
            scale=self._std_dev,
            size=(small_H, small_W),
        )

        # Interpolate the noise map to the original dimensions
        additive_noise = cv2.resize(
            additive_noise,
            (W, H),
            interpolation=cv2.INTER_CUBIC,
        )
        
        # Add the noise to the depth image where the depth is greater than 0
        # and clip the depth values between 0 and the maximum float32 value
        depth[depth > 0] += additive_noise[depth > 0]
        depth = np.clip(depth, 0, np.finfo(np.float32).max)
        
        return depth

class DepthMissingTransform(DepthTransform):
    """
    Randomly drop-out parts of the depth image.
    """
    def __init__(
        self,
        max_missing_fraction: float = 0.2,
        debug: bool = False,
        p: float = 1.0,
    ) -> None:
        """Constructor.

        Args:
            max_missing_fraction (float, optional): Maximum fraction of depth values
                to set to zero. Defaults to 0.2.
            debug (bool, optional): Debug mode to set the fraction of missing depth
                values to a fixed value for testing purposes Defaults to False.
            p (float, optional): Probability of applying the transformation.
                Defaults to 1.0.
        """
        super().__init__(p)
        
        self._max_missing_fraction = max_missing_fraction
        self._debug = debug

    def _transform_depth(self, depth: np.ndarray) -> np.ndarray:
        """Randomly drop-out parts of the depth image.

        Args:
            depth (np.ndarray): Depth image.

        Returns:
            np.ndarray: Depth image with missing values.
        """
        depth = np.copy(depth)
        v_idx, u_idx = np.where(depth > 0)
        
        # Set a random fraction of the depth image to zero
        if not self._debug:
            missing_fraction = np.random.uniform(0, self._max_missing_fraction)
        else:
            missing_fraction = self._max_missing_fraction
        
        # Randomly select the indices to dropout
        dropout_ids = np.random.choice(
            np.arange(len(u_idx)),
            int(missing_fraction * len(u_idx)),
            replace=False,
        )
        
        # Set the corresponding depth values to zero
        depth[v_idx[dropout_ids], u_idx[dropout_ids]] = 0
        
        return depth

class DepthDropoutTransform(DepthTransform):
    """
    Set the entire depth image to zero.
    """
    def _transform_depth(self, depth: np.ndarray) -> np.ndarray:
        """Set the entire depth image to zero.

        Args:
            depth (np.ndarray): Depth image.

        Returns:
            np.ndarray: Depth image with all values set to zero.
        """
        depth = np.zeros_like(depth)
        
        return depth

class DepthEllipseDropoutTransform(DepthTransform):
    """
    Randomly drop a few ellipses in the depth image for robustness.
    """
    def __init__(
        self,
        ellipse_dropout_mean: float = 10.0,
        ellipse_gamma_shape: float = 5.0,
        ellipse_gamma_scale: float = 1.0,
        p: float = 1.0,
    ) -> None:
        """Constructor.

        Args:
            ellipse_dropout_mean (float, optional): Mean number of ellipses to dropout
                (Poisson distribution). Defaults to 10.0.
            ellipse_gamma_shape (float, optional): Shape parameter of the gamma
                distribution for the ellipse radii sampling. Defaults to 5.0.
            ellipse_gamma_scale (float, optional): Scale parameter of the gamma
                distribution for the ellipse radii sampling. Defaults to 1.0.
            p (float, optional): Probability of applying the transformation.
                Defaults to 1.0.
        """
        super().__init__(p)
        
        self._noise_params = {
            "ellipse_dropout_mean": ellipse_dropout_mean,
            "ellipse_gamma_scale": ellipse_gamma_scale,
            "ellipse_gamma_shape": ellipse_gamma_shape,
        }

    @staticmethod
    def generate_random_ellipses(
        depth_img: np.ndarray,
        noise_params: Dict[str, float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate random ellipses to dropout.

        Args:
            depth_img (np.ndarray): Depth image.
            noise_params (Dict[str, float]): Distribution parameters.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Ellipse radii,
                angles, and dropout centers.
        """
        # Sample number of ellipses to dropout
        num_ellipses_to_dropout = np.random.poisson(
            noise_params["ellipse_dropout_mean"],
        )

        # Sample ellipse centers
        nonzero_pixel_indices = np.array(
            np.where(depth_img > 0),
        ).T  # Shape: [#nonzero_pixels x 2]
        dropout_centers_indices = np.random.choice(
            nonzero_pixel_indices.shape[0],
            size=num_ellipses_to_dropout,
        )
        # Shape: [num_ellipses_to_dropout x 2]
        dropout_centers = nonzero_pixel_indices[dropout_centers_indices, :]

        # Sample ellipse radii and angles
        x_radii = np.random.gamma(
            noise_params["ellipse_gamma_shape"],
            noise_params["ellipse_gamma_scale"],
            size=num_ellipses_to_dropout,
        )
        y_radii = np.random.gamma(
            noise_params["ellipse_gamma_shape"],
            noise_params["ellipse_gamma_scale"],
            size=num_ellipses_to_dropout,
        )
        angles = np.random.randint(0, 360, size=num_ellipses_to_dropout)

        return x_radii, y_radii, angles, dropout_centers

    @staticmethod
    def dropout_random_ellipses(
        depth_img: np.ndarray,
        noise_params: Dict[str, float],
    ) -> np.ndarray:
        """Randomly drop a few ellipses in the depth image for robustness.
        Adapted from:
        https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py
        This is adapted from the DexNet 2.0 code:
        https://github.com/chrisdxie/uois/blob/master/src/data_augmentation.py#L53

        Args:
            depth_img (np.ndarray): Depth image.
            noise_params (Dict[str, float]): Distribution parameters.

        Returns:
            np.ndarray: Depth image with missing values.
        """
        depth_img = depth_img.copy()

        # Generate random ellipses to dropout
        (
            x_radii,
            y_radii,
            angles,
            dropout_centers,
        ) = DepthEllipseDropoutTransform.generate_random_ellipses(
            depth_img,
            noise_params=noise_params,
        )

        num_ellipses_to_dropout = x_radii.shape[0]

        # Dropout ellipses
        for i in range(num_ellipses_to_dropout):
            center = dropout_centers[i, :]
            x_radius = np.round(x_radii[i]).astype(int)
            y_radius = np.round(y_radii[i]).astype(int)
            angle = angles[i]

            depth_img = cv2.ellipse(
                depth_img,
                tuple(center[::-1]),
                (x_radius, y_radius),
                angle=angle,
                startAngle=0,
                endAngle=360,
                color=0,  # Depth value set to zero
                thickness=-1,
            )

        return depth_img

    def _transform_depth(self, depth: np.ndarray) -> np.ndarray:
        """Drop a few ellipses in the depth image for robustness.

        Args:
            depth (np.ndarray): Depth image.

        Returns:
            np.ndarray: Depth image with missing values.
        """
        depth = self.dropout_random_ellipses(depth, self._noise_params)
        
        return depth

class DepthEllipseNoiseTransform(DepthTransform):
    """
    Add random Gaussian noise to the depth image in the shape of ellipses.
    """
    def __init__(
        self,
        ellipse_dropout_mean: float = 10.0,
        ellipse_gamma_shape: float = 5.0,
        ellipse_gamma_scale: float = 1.0,
        std_dev: float = 0.01,
        p: float = 1.0,
    ) -> None:
        """Constructor.

        Args:
            ellipse_dropout_mean (float, optional): Mean number of ellipses to dropout.
                Defaults to 10.0.
            ellipse_gamma_shape (float, optional): Shape parameter of the gamma
                distribution for the ellipse radii sampling. Defaults to 5.0.
            ellipse_gamma_scale (float, optional): Scale parameter of the gamma
                distribution for the ellipse radii sampling. Defaults to 1.0.
            std_dev (float, optional): Standard deviation of the Gaussian noise.
                Defaults to 0.01.
            p (float, optional): Probability of applying the transformation.
                Defaults to 1.0.
        """
        super().__init__(p)
        
        self._std_dev = std_dev
        self._noise_params = {
            "ellipse_dropout_mean": ellipse_dropout_mean,
            "ellipse_gamma_scale": ellipse_gamma_scale,
            "ellipse_gamma_shape": ellipse_gamma_shape,
        }

    def _transform_depth(self, depth: np.ndarray) -> np.ndarray:
        """Add random Gaussian noise to the depth image in the shape of ellipses.

        Args:
            depth (np.ndarray): Depth image.

        Returns:
            np.ndarray: Depth image with added noise.
        """
        depth_img = depth
        depth_aug = depth_img.copy()

        # Generate random ellipses for noise
        (
            x_radii,
            y_radii,
            angles,
            dropout_centers,
        ) = DepthEllipseDropoutTransform.generate_random_ellipses(
            depth_img,
            noise_params=self._noise_params,
        )

        num_ellipses_to_dropout = x_radii.shape[0]

        # Sample additive noise
        additive_noise = np.random.normal(
            loc=0.0,
            scale=self._std_dev,
            size=x_radii.shape,
        )

        # Apply additive noise to ellipses
        noise = np.zeros_like(depth)
        
        for i in range(num_ellipses_to_dropout):
            center = dropout_centers[i, :]
            x_radius = np.round(x_radii[i]).astype(int)
            y_radius = np.round(y_radii[i]).astype(int)
            angle = angles[i]

            noise = cv2.ellipse(
                noise,
                tuple(center[::-1]),
                (x_radius, y_radius),
                angle=angle,
                startAngle=0,
                endAngle=360,
                color=additive_noise[i],
                thickness=-1,
            )

        depth_aug[depth > 0] += noise[depth > 0]
        depth = depth_aug

        return depth

class DepthBlurTransform(DepthTransform):
    """
    Blur the depth image.
    """
    def __init__(
        self,
        factor_interval: Tuple[int, int] = (3, 7),
        p: float = 1.0,
    ) -> None:
        """Constructor.

        Args:
            factor_interval (Tuple[int, int], optional): Interval of blur factor.
                Defaults to (3, 7).
            p (float, optional): Probability of applying the transformation.
                Defaults to 1.0.
        """
        super().__init__(p)
        self._factor_interval = factor_interval

    def _transform_depth(self, depth: np.ndarray) -> np.ndarray:
        """Blur the depth image.

        Args:
            depth (np.ndarray): Depth image.

        Returns:
            np.ndarray: Blurred depth image.
        """
        depth = np.copy(depth)
        k = random.randint(*self._factor_interval)
        depth = cv2.blur(depth, (k, k))
        
        return depth


#--------------------------#
# Background augmentations #
#--------------------------#

class DepthBackgroundDropoutTransform(SceneObservationTransform):
    """
    Set all background depth values to zero.
    """
    def _transform(self, obs: SceneObservation) -> SceneObservation:
        """Set all background depth values to zero.

        Args:
            obs (SceneObservation): Scene observation.

        Raises:
            ValueError: If the depth observation is None.
            ValueError: If the segmentation observation is None.

        Returns:
            SceneObservation: Transformed scene observation.
        """
        if obs.depth is None:
            raise ValueError("The depth observation is None.")
        elif obs.segmentation is None:
            raise ValueError("The segmentation observation is None.")

        # Set background to zero
        depth = np.copy(obs.depth)
        depth[obs.segmentation == 0] = 0
        
        return dataclasses.replace(obs, depth=depth)

class BackgroundImageDataset:
    """
    Background image dataset.
    """
    def __init__(self, dataset: Dataset) -> None:
        """Constructor.

        Args:
            dataset (List): A dataset containing images
        """
        self._dataset = dataset
        
    def __getitem__(self, idx: int) -> np.ndarray:
        """Get the background image at the given index.

        Args:
            idx (int): Index of the background image.

        Returns:
            np.ndarray: Background image.
        """
        if isinstance(self._dataset, VOCSegmentation):
            return self._dataset[idx][0]
        else:
            raise ValueError("The background image dataset is not supported.")

    def __len__(self) -> int:
        """Get the number of background images.

        Returns:
            int: Number of background images.
        """
        return len(self._dataset)

class ReplaceBackgroundTransform(SceneObservationTransform):
    """
    Replace the background of the RGB observation with a random image
    from a dataset.
    """
    def __init__(
        self,
        image_dataset: BackgroundImageDataset,
        p: float = 1.0,
    ) -> None:
        """Constructor.

        Args:
            image_dataset (BackgroundImageDataset): Background image dataset.
            p (float, optional): Probability of applying the transformation.
                Defaults to 1.0.
        """
        super().__init__(p)
        
        self._image_dataset = image_dataset

    def _transform(self, obs: SceneObservation) -> SceneObservation:
        """Replace the background of the RGB observation with a random image
        from the dataset.

        Args:
            obs (SceneObservation): Scene observation.

        Raises:
            ValueError: If the RGB observation is None.
            ValueError: If the segmentation observation is None.
            

        Returns:
            SceneObservation: Transformed scene observation.
        """
        if obs.rgb is None:
            raise ValueError("The RGB observation is None.")
        elif obs.segmentation is None:
            raise ValueError("The segmentation observation is None.")
        
        rgb = obs.rgb.copy()
        h, w, _ = rgb.shape
        
        # Get a random background image
        rgb_bg_pil =\
            self._image_dataset[random.randint(0, len(self._image_dataset) - 1)]
        rgb_bg = np.asarray(rgb_bg_pil.resize((w, h)))
        
        # Background mask
        mask_bg = obs.segmentation == 0
        # Replace the background with the random image
        rgb[mask_bg] = rgb_bg[mask_bg]
        
        return dataclasses.replace(obs, rgb=rgb)

class VOCBackgroundAugmentation(ReplaceBackgroundTransform):
    """
    Augmentation that replaces the background with a random image from the
    VOC 2012 dataset.
    """
    def __init__(self, voc_root: Path, p: float = 1.0) -> None:
        """Constructor.

        Args:
            voc_root (Path): Path to the VOC 2012 dataset.
            p (float, optional): Probability of applying the transformation.
                Defaults to 1.0.
        """
        # Load the VOC 2012 dataset
        voc_dataset = VOCSegmentation(
            root=voc_root,
            year="2012",
            image_set="trainval",
            download=False,
        )
        
        image_dataset = BackgroundImageDataset(voc_dataset)
        
        super().__init__(image_dataset, p)


#------------------------#
# Observation transforms #
#------------------------#

class CropResizeToAspectTransform(SceneObservationTransform):
    """
    Crop and resize the RGB, segmentation, and depth observations to a target
    aspect ratio.
    """
    def __init__(self, resize: Resolution = (480, 640), p: float = 1.0) -> None:
        """Constructor.

        Args:
            resize (Resolution, optional): Target aspect ratio (height, width).
                Defaults to (480, 640).
            p (float, optional): Probability of applying the transformation.
                Defaults to 1.0.

        Raises:
            ValueError: If the width is less than the height.
        """
        super().__init__(p)
        
        if resize[1] < resize[0]:
            raise ValueError("The width must be greater than the height.")
        
        self._resize = resize
        self._aspect = max(resize) / min(resize)
    
    @staticmethod
    def make_detections_from_segmentation(
        segmentations: np.ndarray,
    ) -> List[Dict[int, np.ndarray]]:
        """Make detections from segmentations.

        Args:
            segmentations (np.ndarray): Segmentations.

        Returns:
            List[Dict[int, np.ndarray]]: List of detections.
        """
        assert segmentations.ndim == 3
        detections = []
        
        for segmentation_n in segmentations:
            
            dets_n = {}
            
            for unique_id in np.unique(segmentation_n):
                
                ids = np.where(segmentation_n == unique_id)
                x1, y1, x2, y2 = (
                    np.min(ids[1]),
                    np.min(ids[0]),
                    np.max(ids[1]),
                    np.max(ids[0]),
                )
                dets_n[int(unique_id)] = np.array([x1, y1, x2, y2])
            
            detections.append(dets_n)
        
        return detections

    def _transform(self, obs: SceneObservation) -> SceneObservation:
        """Crop and resize the RGB, segmentation, and depth observations to a
        target aspect ratio.

        Args:
            obs (SceneObservation): Scene observation.

        Raises:
            ValueError: If the RGB observation is None.
            ValueError: If the segmentation observation is None.
            ValueError: If the binary masks are None.
            ValueError: If the camera data is None.
            ValueError: If the object datas are None.
            ValueError: If the segmentation dtype is not uint32.
            ValueError: If the depth dtype is not float32.
            ValueError: If the segmentation mode is not 'I'.
            ValueError: If the depth mode is not 'F'.

        Returns:
            SceneObservation: Transformed scene observation.
        """
        if obs.rgb is None:
            raise ValueError("The RGB observation is None.")
        elif obs.segmentation is None:
            raise ValueError("The segmentation observation is None.")
        elif obs.binary_masks is None:
            raise ValueError("The binary masks are None.")
        elif obs.camera_data is None:
            raise ValueError("The camera data is None.")
        elif obs.object_datas is None:
            raise ValueError("The object datas are None.")
        elif obs.segmentation.dtype != np.uint32:
            raise ValueError("The segmentation dtype is not uint32.")

        rgb_pil = PIL.Image.fromarray(obs.rgb)
        w, h = rgb_pil.size

        # Skip if the image is already at the target size
        if (h, w) == self._resize:
            return obs

        segmentation_pil = PIL.Image.fromarray(obs.segmentation)
        
        if segmentation_pil.mode != "I":
            raise ValueError("The segmentation mode is not 'I'.")
        
        depth_pil = None
        
        if obs.depth is not None:
            if obs.depth.dtype != np.float32:
                raise ValueError("The depth dtype is not float32.")
            
            depth_pil = PIL.Image.fromarray(obs.depth)
            
            if depth_pil.mode != "F":
                raise ValueError("The depth mode is not 'F'.")

        # Match the width on input image with an image of target aspect ratio.
        if not np.isclose(w / h, self.aspect):
            r = self._aspect
            crop_h = w * 1 / r
            x0, y0 = w / 2, h / 2
            crop_box_size = (crop_h, w)
            crop_h, crop_w = min(crop_box_size), max(crop_box_size)
            x1, y1, x2, y2 = (
                x0 - crop_w / 2,
                y0 - crop_h / 2,
                x0 + crop_w / 2,
                y0 + crop_h / 2,
            )
            box = (x1, y1, x2, y2)
            rgb_pil = rgb_pil.crop(box)
            segmentation_pil = segmentation_pil.crop(box)
            
            if depth_pil is not None:
                depth_pil = depth_pil.crop(box)
            
            new_K = get_K_crop_resize(
                torch.tensor(obs.camera_data.K).unsqueeze(0),
                torch.tensor(box).unsqueeze(0),
                orig_size=(h, w),
                crop_resize=(crop_h, crop_w),
            )[0].numpy()
        
        else:
            new_K = obs.camera_data.K

        # Resize to target size
        w, h = rgb_pil.size
        w_resize, h_resize = max(self._resize), min(self._resize)
        rgb_pil = rgb_pil.resize((w_resize, h_resize), resample=PIL.Image.BILINEAR)
        segmentation_pil = segmentation_pil.resize(
            (w_resize, h_resize),
            resample=PIL.Image.NEAREST,
        )
        
        if depth_pil is not None:
            depth_pil = depth_pil.resize(
                (w_resize, h_resize),
                resample=PIL.Image.NEAREST,
            )
        
        box = (0, 0, w, h)
        new_K = get_K_crop_resize(
            torch.tensor(new_K).unsqueeze(0),
            torch.tensor(box).unsqueeze(0),
            orig_size=(h, w),
            crop_resize=(h_resize, w_resize),
        )[0].numpy()

        new_obs = deepcopy(obs)
        new_obs.camera_data.K = new_K
        new_obs.camera_data.resolution = (h_resize, w_resize)
        new_obs.rgb = np.array(rgb_pil, dtype=np.uint8)
        new_obs.segmentation = np.array(segmentation_pil, dtype=np.int32)
        
        if depth_pil is not None:
            new_obs.depth = np.array(depth_pil, dtype=np.float_)

        # Update modal object bounding boxes
        dets_gt =\
            CropResizeToAspectTransform.make_detections_from_segmentation(
                new_obs.segmentation[None]
            )[0]
        
        new_object_datas = []
        
        for obj in obs.object_datas:
            
            if obj.unique_id in dets_gt:
                new_obj = dataclasses.replace(
                    obj,
                    bbox_modal=dets_gt[obj.unique_id],
                    bbox_amodal=None,
                    visib_fract=None,
                )
                new_object_datas.append(new_obj)
        
        new_obs.object_datas = new_object_datas
        
        return new_obs
