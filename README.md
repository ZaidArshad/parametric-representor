# Recursive Superquadric Shape Fitting with EMS

## About

This is a project for our course, `CMPT 464: Geometric Modeling in Computer Graphics`. The source code for our paper `Recursive Superquadric Shape Fitting with EMS` can be found here.

### Authors

Arlo Watts
Jordan McKenzie
Zaid Arshad

## Setup

We have included scripts to setup a Python virtual environment and install the project's dependencies.

### Windows
`setup.bat`

### Unix
`source setup.sh`

To run the model once on a single shape, run `python src/main.py model_name`, where `model_name` is `bunny`, `chair`, `dog`, `hand`, `pot`, `rod`, `sofa`, or `statue`. The script will look for `data/model_name/surface_points.ply`.

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

Weixiao Liu et al. “Robust and Accurate Superquadric Recovery: a Probabilistic Approach”. In: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2022, pp. 2666–2675. doi: 10.1109/CVPR52688.2022.00270.

Simsangcheol. Chamfer Distance. 2023. url: <https://medium.com/@sim30217/chamfer-distance-4207955e8612>

Wikipedia. Flood fill. url: <https://en.wikipedia.org/wiki/Flood_fill>

Zekun Hao, Hadar Averbuch-Elor, Noah Snavely, Serge Belongie. DualSDF. 2020. url: <https://github.com/zekunhao1995/DualSDF>

Despoina Paschalidou, Ali Osman Ulusoy, and Andreas Geiger. “Superquadrics Revisited: Learning 3D Shape Parsing beyond Cuboids”. In: Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR). 2019.

Shubham Tulsiani et al. Learning Shape Abstractions by Assembling Volumetric Primitives. 2018. arXiv: 1612 . 00404 [cs.CV]. url: <https://arxiv.org/abs/1612.00404>.

Weixiao Liu et al. “Marching-Primitives: Shape Abstraction from Signed Distance Function”. In: Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR). 2023.

Barr. “Superquadrics and Angle-Preserving Transformations”. In: IEEE Computer Graphics and Applications 1.1 (1981), pp. 11–23. doi: 10.1109/MCG.1981.1673799.

A.D. Gross and T.E. Boult. “Error Of Fit Measures For Recovering Parametric Solids”. In: [1988 Proceedings] Second International Conference on Computer Vision. 1988, pp. 690–694. doi: 10.1109/CCV.1988.590052.

Anil K. Jain. “Data clustering: 50 years beyond K-means”. In: Pattern Recognition Letters 31.8 (2010). Award winning papers from the 19th International Conference on Pattern Recognition (ICPR), pp. 651–666. issn: 0167-8655. doi: <https://doi.org/10.1016/j.patrec.2009.09.011>. url: <https://www.sciencedirect.com/science/article/pii/S0167865509002323>.

Jasper Snoek, Hugo Larochelle, and Ryan P. Adams. Practical Bayesian Optimization of Machine Learning Algorithms. 2012. arXiv: 1206.2944 [stat.ML]. url: <https://arxiv.org/abs/1206.2944>.

Ainesh Bakshi et al. A Near-Linear Time Algorithm for the Chamfer Distance. 2023. arXiv: 2307.03043 [cs.DS]. url: <https://arxiv.org/abs/2307.03043>.

Ayushi Sharma. From Point Clouds to Voxel Grids: A Practical Guide to 3D Data Voxelization. 2025. url: <https://medium.com/@ayushi.sharma.3536/from-point-clouds-to-voxel-grids-a-practical-guide-to-3d-data-voxelization-cf5991c1e7bb>.
