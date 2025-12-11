# parametric-representator
## About
This is a project for our course, `CMPT 464: Geometric Modeling in Computer Graphics`. The source code for our paper `Recursive Superquadric Shape Fitting with EMS` can be found here.

### Authors
    Arlo Watts
    Jordan McKenzie
    Zaid Arshad

## Setup
### Windows
`setup.bat`

### Unix
`source setup.sh` 

## Data
The models that can be found in the `data` directory come in the form of meshes, SDFs, voxels and point clouds. For reference: `.npz` file contain keys `[voxels, sdf_points, sdf_values, centroid, scale]`

## Running tests
### Options
    test_type (string): Which fitting algorithms you want to test.
        all:            Test both baseline and superquadric fitting algorithms.
        base:           Test the baseline algorithm, SphereNet only.
        superquadric:   Test the superquadric fitting algorithm only.

    save_to_file (int): Whether to saves quantitative results to the results directory in the form of csv files.
        0:              Do not save results to file.
        1:              Save results to results/*.csv and summary to results/summary.csv

    show_visual (int): Whether to show the trimesh visual for each input model.
        0:             Do not show the visual for any model.
        1:             Show the visual in a popped out window for fitted model after they run. 

    iterations (int): Number of iterations to run on each model of the given algorithm(s)

    model_name (list): Space separated list of each model to test the given algorithm(s) against. 

### Command
`src/results.py` is the script that runs in order to run the tests.

Example: `python src/results.py all 0 1 5 dog hand pot` will run both fitting algorithms, will NOT save the results to file, will show the visual for each model fitting, run 5 iterations, and test the fitting of the dog, hand, and pot models independently. 

## References
        Simsangcheol, Chamfer distance, URL: https://medium.com/@sim30217/chamfer-distance-4207955e8612

        Ayushi Sharma, From Point Clouds to Voxel Grids: A Practical Guide to 3D Data Voxelization, URL: https://medium.com/@ayushi.sharma.3536/from-point-clouds-to-voxel-grids-a-practical-guide-to-3d-data-voxelization-cf5991c1e7bb

        Wikipedia, Flood fill, URL: https://en.wikipedia.org/wiki/Flood_fill

        Zekun Hao,  Hadar Averbuch-Elor,  Noah Snavely, SergeBelongie, DualSDF, URL: https://github.com/zekunhao1995/DualSDF 
