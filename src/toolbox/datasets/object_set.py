"""
This module contains classes to represent rigid objects and sets of rigid objects.
"""
from __future__ import annotations  # Required for forward references

# Standard libraries
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union


class RigidObject:
    """
    A class to represent a rigid object. It gathers information about the object
    (e.g., label, path to the mesh, diameter, etc.).
    """
    def __init__(
        self,
        label: str,
        mesh_path: Path,
        mesh_diameter: Optional[float] = None,
        scaling_factor_mesh_units_to_meters: float = 1.0,
        scaling_factor: float = 1.0,
        ypr_offset_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        ) -> None:
        """Initializes a rigid object.

        Args:
            label (str): A unique label to identify an object.
            mesh_path (Path): Path to a mesh. Multiple object types are supported.
                Please refer to downstream usage of this class for the supported
                formats.
            mesh_diameter (Optional[float], optional): Diameter of the object, expressed
                in the unit of the meshes. Defaults to None.
            scaling_factor_mesh_units_to_meters (float, optional): Scale that converts
                mesh units to meters. Defaults to 1.0.
            scaling_factor (float, optional): An extra scaling factor that can be
                applied to the mesh to rescale it. Defaults to 1.0.
            ypr_offset_deg (Tuple[float, float, float], optional): A rotation offset
                applied to the mesh. This can be useful to correct some mesh conventions
                where axes are flipped. Defaults to (0.0, 0.0, 0.0).
        """
        # Assign the object properties
        self._label = label
        self._mesh_path = mesh_path
        self._scaling_factor_mesh_units_to_meters =\
            scaling_factor_mesh_units_to_meters
        self._scaling_factor = scaling_factor
        self._mesh_diameter = mesh_diameter
        self._ypr_offset_deg = ypr_offset_deg
    
    @property
    def label(self) -> str:
        """Returns the label of the object.

        Returns:
            str: The label of the object.
        """
        return self._label

    @property
    def scale(self) -> float:
        """
        Scale factor that converts the mesh to desired units.
        
        Returns:
            float: The scale factor.
        """
        return self._scaling_factor_mesh_units_to_meters * self._scaling_factor
    
    @property
    def mesh_path(self) -> Path:
        """Returns the path to the mesh.

        Returns:
            Path: The path to the mesh.
        """
        return self._mesh_path
    
    @property
    def mesh_diameter(self) -> Optional[float]:
        """Returns the diameter of the object.

        Returns:
            Optional[float]: The diameter of the object.
        """
        return self._mesh_diameter


class RigidObjectSet:
    """
    A class to represent a set of rigid objects.
    """
    def __init__(
        self,
        objects: List[RigidObject],
        ) -> None:
        """Initializes a set of rigid objects.

        Args:
            objects (List[RigidObject]): A list of rigid objects.

        Raises:
            RuntimeError: If there are objects with duplicate labels.
        """
        # Assign the object properties
        self._objects = objects
        self._labels_to_objects = {obj._label: obj for obj in objects}
        
        # Check for duplicate labels
        if len(self._labels_to_objects) != len(objects):
            raise RuntimeError("There are objects with duplicate labels.")
        
        # Define a common scale if all objects share the same scaling factor
        self._common_scale = objects[0].scale
        for obj in objects:
            if obj.scale != self._common_scale:
                self._common_scale = None
                break
        
    def __getitem__(self, id: Union[int, str]) -> RigidObject:
        """Returns an object by its index or label.

        Args:
            id (Union[int, str]): Index or label of the object.
        
        Returns:
            RigidObject: The object corresponding to the index or label.
        
        Raises:
            ValueError: If the type of the index is invalid.
        """
        if isinstance(id, int):
            return self._objects[id]
        elif isinstance(id, str):
            return self._labels_to_objects[id]
        else:
            raise ValueError("Invalid type for id. Must be int or str.")
    
    def __len__(self) -> int:
        """Returns the number of objects in the set.

        Returns:
            int: The number of objects in the set.
        """
        return len(self._objects)

    @property
    def objects(self) -> List[RigidObject]:
        """Returns the list of objects.

        Returns:
            List[RigidObject]: The list of objects.
        """
        return self._objects
    
    @property
    def scale(self) -> float:
        """
        Scale factor that converts the meshes to desired units.
        
        Returns:
            float: The scale factor.
        """
        return self._common_scale
    
    def get_id_from_label(self, label: str) -> int:
        """Returns the index of an object by its label.

        Args:
            label (str): The label of the object.

        Returns:
            int: The index of the object.
        """
        return self._objects.index(self._labels_to_objects[label])
    
    def filter_objects(self, keep_labels: Set[str]) -> RigidObjectSet:
        """Filters the objects by keeping only the objects with the specified labels.

        Args:
            keep_labels (Set[str]): The labels of the objects to keep.

        Returns:
            RigidObjectSet: A new object set with the filtered objects.
        """
        objects_to_keep = [obj for obj in self._objects if obj.label in keep_labels]
        return RigidObjectSet(objects_to_keep)
