#!/bin/bash

SOURCE_FOLDER="./June-5"
DESTINATION_FOLDER="./All_Data/June-5/retrain/imgs"

mkdir -p "$DESTINATION_FOLDER"

for subfolder in "$SOURCE_FOLDER"/*/; do

    if [ -d "$subfolder/images" ]; then

        mv "$subfolder/images"/* "$DESTINATION_FOLDER"
        
        rmdir "$subfolder/images"
        rmdir "$subfolder"
        
        echo "Moved images from $subfolder/images to $DESTINATION_FOLDER"
    fi
done

rmdir "$SOURCE_FOLDER"