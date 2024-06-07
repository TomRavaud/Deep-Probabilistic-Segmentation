# Standard libraries
from pathlib import Path
from typing import List

# Custom modules
from toolbox.datasets.object_set import RigidObject, RigidObjectSet


class BCOTObjectSet(RigidObjectSet):
    """
    A class to represent a set of BCOT objects.
    """
    def __init__(self, bcot_root: Path) -> None:
        """Initializes a set of BCOT objects.

        Args:
            bcot_root (Path): Root directory of the BCOT objects.
            split (str, optional): Split of the set of objects. Defaults to "orig".
        """
        # Set the directory for the GSO models
        self.bcot_dir = bcot_root

        scaling_factor = 1.0

        # Get the list of valid object IDs
        object_ids = BCOTObjectSet._get_valid_object_ids(self.bcot_dir)
        
        # Create a list of RigidObject objects
        objects = []
        
        for object_id in object_ids:
            
            model_path = self.bcot_dir / object_id
            label = f"{object_id.split('.')[0]}"
            
            obj = RigidObject(
                label=label,
                mesh_path=model_path,
                scaling_factor=scaling_factor,
                mesh_diameter=1.0,
            )
            objects.append(obj)
        
        # Initialize the parent class
        super().__init__(objects)
    
    
    @staticmethod
    def _get_valid_object_ids(
        objset_dir: Path,
        ) -> List[str]:
        """Returns a list of valid object IDs in the object set directory.

        Args:
            objset_dir (Path): Directory containing the object set.

        Returns:
            List[str]: List of valid object IDs.
        """
        # Get the list of object models
        models_files = objset_dir.iterdir()
        
        # Create a list of object IDs
        object_ids = []

        for model_file in models_files:
            if (model_file).exists():
                # Append the object ID to the list
                object_ids.append(model_file.name)

        object_ids.sort()

        return object_ids
