import results
import argparse
import trimesh

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name')
    args = parser.parse_args()

    points = trimesh.load(f"data/{args.model_name}/surface_points.ply").vertices

    results.superquadric_test(points)
