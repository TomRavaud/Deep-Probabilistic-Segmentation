# Standard libraries
from pathlib import Path
import json
from typing import List

# Custom modules
from toolbox.datasets.object_set import RigidObject, RigidObjectSet


class GoogleScannedObjectSet(RigidObjectSet):
    """
    A class to represent a set of Google Scanned Objects.
    """
    def __init__(self, gso_root: Path, split: str = "orig") -> None:
        """Initializes a set of Google Scanned Objects.

        Args:
            gso_root (Path): Root directory of the Google Scanned Objects.
            split (str, optional): Split of the set of objects. Defaults to "orig".
        """
        # Set the directory for the GSO models
        self.gso_dir = gso_root / f"models_{split}"

        # Set the scaling factor based on the split
        if split == "orig":
            scaling_factor = 30.0
        elif split in {"normalized", "pointcloud"}:
            scaling_factor = 0.1

        # Get the list of valid object IDs
        object_ids = GoogleScannedObjectSet._get_valid_object_ids(self.gso_dir)
        
        # Create a list of RigidObject objects
        objects = []
        
        for object_id in object_ids:
            
            model_path = self.gso_dir / object_id / "meshes" / "model.obj"
            label = f"gso_{object_id}"
            
            obj = RigidObject(
                label=label,
                mesh_path=model_path,
                scaling_factor=scaling_factor,
            )
            objects.append(obj)
        
        # Initialize the parent class
        super().__init__(objects)
    
    
    @staticmethod
    def _get_valid_object_ids(
        objset_dir: Path,
        model_name: str = "model.obj",
        ) -> List[str]:
        """Returns a list of valid object IDs in the object set directory.

        Args:
            objset_dir (Path): Directory containing the object set.
            model_name (str, optional): Name of the model file. Defaults to "model.obj".

        Returns:
            List[str]: List of valid object IDs.
        """
        # Get the list of object directories
        models_dir = objset_dir.iterdir()
        
        # Get the list of invalid object IDs
        invalid_ids =\
            set(json.loads((objset_dir.parent / "invalid_meshes.json").read_text()))
        
        # Create a list of object IDs
        object_ids = []

        for model_dir in models_dir:
            if model_dir.name not in invalid_ids and\
                (model_dir / "meshes" / model_name).exists():
                # Append the object ID to the list
                object_ids.append(model_dir.name)

        object_ids.sort()

        return object_ids
