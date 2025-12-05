# parametric-representator
CMPT 464 final group project.

## Setup
### Windows
`setup.bat`

### Unix
`source setup.sh` 

## Data
For reference: `.npz` file contain keys `[voxels, sdf_points, sdf_values, centroid, scale]`

## Running tests
`python src/results.py test_type:[all | base | superquadric] save_to_file:[0 | 1] show_visual:[0 | 1] iterations:[number] model_name:[space separated list]`

Example `python src/results.py all 1 0 5 dog hand pot`