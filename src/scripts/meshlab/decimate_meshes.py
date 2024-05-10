"""
This script decimates a set a meshes using MeshLab's quadric edge collapse decimation
algorithm.
"""
# Standard libraries
import pathlib
import argparse

# Third-party libraries
import pymeshlab
from tqdm import tqdm


# Create an argument parser
parser = argparse.ArgumentParser(
    description="Decimate a set of meshes using MeshLab's quadric edge collapse "
    "decimation algorithm."
)
parser.add_argument(
    "--src_object_set_path",
    type=str,
    default="data/webdatasets/google_scanned_objects/models_normalized/",
    help="Path to the source object set directory.",
)
parser.add_argument(
    "--target_object_set_path",
    type=str,
    default="data/webdatasets/google_scanned_objects/models_normalized_decimated/",
    help="Path to the target object set directory.",
)
parser.add_argument(
    "--num_faces",
    type=int,
    default=1000,
    help="Target number of faces in the decimated mesh.",
)

# Parse the arguments
args = parser.parse_args()


# Create a new MeshSet object
ms = pymeshlab.MeshSet()

# Set the decimation configuration
decimation_cfg = {
    "targetfacenum": args.num_faces,
    "preserveboundary": True,
}

# Source and target object set paths
src_object_set_path = pathlib.Path(args.src_object_set_path)
target_object_set_path = pathlib.Path(args.target_object_set_path)

# Create the target object set directory if it does not exist
if not target_object_set_path.exists():
    target_object_set_path.mkdir(parents=True, exist_ok=True)

# Get the list of all the object directories
object_directories = [x for x in src_object_set_path.iterdir() if x.is_dir()]


# Iterate over all the object directories
for object_directory in tqdm(object_directories):
    
    if (target_object_set_path / object_directory.name / "meshes").exists():
        continue
        
    # Create the target directory if it does not exist
    (target_object_set_path / object_directory.name / "meshes").mkdir(
        parents=True, exist_ok=True)

    # Get the mesh file in the object directory
    mesh_file = src_object_set_path / object_directory.name / "meshes/model.obj"
    
    # Load the mesh file
    ms.load_new_mesh(str(mesh_file))
    
    # Decimate the mesh
    ms.meshing_decimation_quadric_edge_collapse_with_texture(**decimation_cfg)
    
    # Save the decimated mesh
    ms.save_current_mesh(
        str(target_object_set_path / object_directory.name / "meshes" / "model.obj")
    )
