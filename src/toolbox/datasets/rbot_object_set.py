# Standard libraries
from pathlib import Path
from typing import List

# Custom modules
from toolbox.datasets.object_set import RigidObject, RigidObjectSet


class RBOTObjectSet(RigidObjectSet):
    """
    A class to represent a set of RBOT objects.
    """
    def __init__(self, rbot_root: Path) -> None:
        """Initializes a set of RBOT objects.

        Args:
            rbot_root (Path): Root directory of the RBOT objects.
            split (str, optional): Split of the set of objects. Defaults to "orig".
        """
        # Set the directory for the GSO models
        self.rbot_dir = rbot_root

        scaling_factor = 1.0

        # Get the list of valid object IDs
        object_ids = RBOTObjectSet._get_valid_object_ids(self.rbot_dir)
        
        # Create a list of RigidObject objects
        objects = []
        
        for object_id in object_ids:
            
            model_path = self.rbot_dir / object_id / f"{object_id}.obj"
            
            label = object_id
            
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
        # Get the list of object models directories
        models_dir = [d.name for d in objset_dir.iterdir() if d.is_dir()]
        
        return models_dir
        