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
            mesh_units (str, optional): Units in which the vertex positions are
                expressed. Can be "m" or "mm", defaults to "m". In the operations of
                this codebase, all mesh coordinates and poses must be expressed in
                meters. When an object is loaded, a scaling will be applied to the mesh
                to ensure its coordinates are in meters when in memory. Defaults to "m".
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
    def scale(self) -> float:
        """
        Scale factor that converts the mesh to desired units.
        
        Returns:
            float: The scale factor.
        """
        return self._scaling_factor_mesh_units_to_meters * self._scaling_factor


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

    def filter_objects(self, keep_labels: Set[str]):
        """Filters the objects by keeping only the objects with the specified labels.

        Args:
            keep_labels (Set[str]): The labels of the objects to keep.
        """
        self._objects = [obj for obj in self._objects if obj.label in keep_labels]
        # Update the labels to objects mapping
        self._labels_to_objects = {obj.label: obj for obj in self._objects}
