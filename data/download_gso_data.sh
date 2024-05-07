#!/bin/bash

# Create a directory to store the dataset if it does not exist
if [ -d "webdatasets" ]; then
    echo "Info: 'webdatasets' directory already exists. Writing inside it."
else
    mkdir -p webdatasets
fi
cd webdatasets

# Function to download shards of the GSO dataset
download_gso() {

    # Create a directory to store the dataset if it does not exist
    if [ -d "gso_1M" ]; then
        echo "Info: 'gso_1M' directory already exists."
        # Ask the user if he wants to give another name, overwrite the existing directory or exit
        read -p 'Do you want to overwrite the existing directory? (yes/no): ' overwrite
        if [[ $overwrite == "yes" ]]; then
            rm -rf gso_1M
            mkdir -p gso_1M
            cd gso_1M
        else
            read -p 'Do you want to give another name to the directory? (yes/no): ' use_another_name
            if [[ $use_another_name == "yes" ]]; then
                read -p 'Enter the new name: ' new_name
                mkdir -p $new_name
                cd $new_name
            else
                echo "Info: Abandoning the download of the GSO dataset."
                return
            fi
        fi
    fi

    # Get the number of shards to download
    read -p 'Number of shards to download (default/max=1050): ' nb_shards
    nb_shards=${nb_shards:-1050}  # Default value if the input is empty
    nb_shards=$((nb_shards - 1))

    # Download the shards
    for i in $(seq -f "%04g" 0 $nb_shards); do
        wget "https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/webdatasets/gso_1M/0000$i.tar"
    done

    # Download additional files
    wget 'https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/webdatasets/gso_1M/frame_index.feather'
    wget 'https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/webdatasets/gso_1M/infos.json'

    cd ..
}

read -p 'Do you want to download the GSO dataset? (yes/no): ' download_gso_shards

if [[ $download_gso_shards == "yes" ]]; then
    download_gso
fi

read -p 'Do you want to download the 3D models? (yes/no): ' download_gso_models

if [[ $download_gso_models == "yes" ]]; then
    # Download the 3D models
    wget 'https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/tars/google_scanned_objects.zip'
    # Unzip the 3D models
    unzip google_scanned_objects.zip
fi
