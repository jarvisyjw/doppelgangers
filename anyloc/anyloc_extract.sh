#!/bin/bash

echo "Extracting Pittsburgh250k"
echo "Extracting database sequence 005"
python anyloc_mini.py --in_dir=data/Pittsburgh250k/database/005 \
                          --gdesc_dir=data/Pittsburgh250k/gdesc/005

echo "Extracting database sequence 006"
python anyloc_mini.py --in_dir=data/Pittsburgh250k/database/006 \
                          --gdesc_dir=data/Pittsburgh250k/gdesc/006