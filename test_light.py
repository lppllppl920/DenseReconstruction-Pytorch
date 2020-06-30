import cv2
import numpy as np
from pathlib import Path
import argparse
import h5py
import tqdm

# Local
import cg_utils
import models


def display_depth_map(depth_map, min_value=None, max_value=None, colormode=cv2.COLORMAP_JET, scale=None):
    if (min_value is None or max_value is None) and scale is None:
        if len(depth_map[depth_map > 0]) > 0:
            min_value = np.min(depth_map[depth_map > 0])
        else:
            min_value = 0.0
    elif scale is not None:
        min_value = 0.0
        max_value = scale
    else:
        pass

    depth_map_visualize = np.abs((depth_map - min_value) / (max_value - min_value) * 255)
    depth_map_visualize[depth_map_visualize > 255] = 255
    depth_map_visualize[depth_map_visualize <= 0.0] = 0
    depth_map_visualize = cv2.applyColorMap(np.uint8(depth_map_visualize), colormode)
    return depth_map_visualize


def surface_mesh_global_scale(surface_mesh):
    max_bound = np.max(surface_mesh.vertices, axis=0)
    min_bound = np.min(surface_mesh.vertices, axis=0)

    return np.linalg.norm(max_bound - min_bound, ord=2), np.linalg.norm(min_bound, ord=2), np.abs(
        max_bound[2] - min_bound[0])


def main():
    cv2.destroyAllWindows()
    parser = argparse.ArgumentParser(
        description='Depth fusion and surface reconstruction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_root', type=str, required=True,
                        help='root storing the data')
    parser.add_argument('--visualize_fused_model', action='store_true',
                        help='whether or not to visualize fused 3d model')
    parser.add_argument('--trunc_margin_multiplier', type=float, default=10.0,
                        help='truncate margin factor of the signed distance function')
    parser.add_argument('--sequence_root', type=str, default=None,
                        help='root of one video sequence')
    parser.add_argument('--id_list', nargs='+', type=int, required=True,
                        help='list of patient id')
    parser.add_argument('--max_voxel_count', type=float, default=400.0 ** 3,
                        help='maximum count of voxels for depth map fusion')
    args = parser.parse_args()

    folder_list = list(Path(args.data_root).rglob("_start*"))
    folder_list.sort()

    for patient_id in args.id_list:
        data_root = Path(args.data_root) / "{}".format(patient_id)
        sub_folders = list(data_root.glob("_start*/"))
        sub_folders.sort()
        for folder in sub_folders:
            print("Processing {}...".format(str(folder)))
            if args.sequence_root is not None:
                if str(folder) != args.sequence_root:
                    continue

            # Read hdf5 file
            hdf5_path = folder / "fusion_data.hdf5"
            if not hdf5_path.exists():
                print("{} not exists".format(str(hdf5_path)))
                continue
            fusion_data = h5py.File(str(hdf5_path), 'r', libver='latest')

            print("Estimating voxel volume bounds...")
            vol_bnds = np.zeros((3, 2))

            fusion_data_mask_array = fusion_data["mask"]
            _, height, width, _ = fusion_data_mask_array.shape
            mask_boundary = fusion_data_mask_array[0].astype(np.float32).reshape((height, width, 1))

            fusion_data_mean_depth = fusion_data["mean_depth"]
            fusion_data_std_depth = fusion_data["std_depth"]
            fusion_data_extrinsics = fusion_data["extrinsics"]
            fusion_data_intrinsics = fusion_data["intrinsics"]
            fusion_data_color = fusion_data["color"]

            n_imgs = fusion_data_mean_depth.shape[0]
            cam_pose_list = []
            cam_intr = fusion_data_intrinsics[0, :, :].reshape((3, 3))
            for i in range(n_imgs):
                mean_depth_im = fusion_data_mean_depth[i, :, :, :].reshape((height, width, 1))
                cam_pose_list.append(fusion_data_extrinsics[i, :, :].reshape((4, 4)))
                # Compute camera view frustum and extend convex hull
                view_frust_pts = models.get_view_frustum(mean_depth_im, cam_intr, cam_pose_list[i])
                vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
                vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))

            # Avoid nan sample
            if np.any(np.isnan(np.asarray(cam_pose_list))):
                print("NAN sequence encountered in {}".format(folder))
                continue

            voxel_size = 0.1
            vol_dim = (vol_bnds[:, 1] - vol_bnds[:, 0]) / voxel_size
            # Adaptively change the size of one voxel to fit into the GPU memory
            volume = vol_dim[0] * vol_dim[1] * vol_dim[2]
            factor = (volume / args.max_voxel_count) ** (1.0 / 3.0)
            voxel_size *= factor
            print("voxel size: {}".format(voxel_size))

            fused_model_path = str(folder / "fused_mesh.ply")
            scene, surface_mesh = cg_utils.load_3d_model(fused_model_path)

            mesh_global_scale, min_distance, z_distance = surface_mesh_global_scale(surface_mesh)
            print("Fused mesh global scale: {}".format(mesh_global_scale))

            rendering_color_image_list = []
            depth_image_list = []
            print("Generating simulation data...")
            tq = tqdm.tqdm(total=len(n_imgs))
            tq.set_description('Rendering')
            for i in range(n_imgs):
                # 4x4 rigid transformation matrix T^(world)_(camera)
                print("Rendering frame {}...".format(i))
                cam_pose = cam_pose_list[i]
                rendering_color_image, depth_image = cg_utils.get_depth_image_from_3d_model(scene, cam_intr,
                                                                                            height, width, cam_pose,
                                                                                            z_near=1.0e-8,
                                                                                            z_far=mesh_global_scale,
                                                                                            point_light_strength=2.0 * voxel_size,
                                                                                            ambient_strength=0.8)

                rendering_color_image = cv2.cvtColor(rendering_color_image, cv2.COLOR_RGBA2RGB)
                rendering_color_image_list.append(
                    (rendering_color_image * mask_boundary).reshape((height, width, 3)).astype(np.uint8))

                mask_image = (depth_image > 0.0).astype(np.float32)
                depth_image_list.append(mask_image * depth_image)
                tq.update(1)

            max_depth = np.max(np.asarray(depth_image_list))
            max_std_depth = np.max(fusion_data_std_depth)
            GIF_image_list = []

            for i, simulated_depth_image in enumerate(depth_image_list):
                display_simulated_depth_image = display_depth_map(depth_map=simulated_depth_image * mask_boundary.
                                                                  reshape((height, width)),
                                                                  colormode=cv2.COLORMAP_JET, scale=max_depth)
                display_simulated_depth_image = cv2.cvtColor(display_simulated_depth_image, cv2.COLOR_BGR2RGB)

                predicted_mean_depth_image = fusion_data_mean_depth[i, :, :, :].reshape((height, width, 1))
                display_predicted_depth_image = display_depth_map(depth_map=predicted_mean_depth_image,
                                                                  colormode=cv2.COLORMAP_JET, scale=max_depth)
                display_predicted_depth_image = cv2.cvtColor(display_predicted_depth_image, cv2.COLOR_BGR2RGB)

                predicted_std_depth_image = fusion_data_std_depth[i, :, :, :].reshape((height, width, 1))
                display_predicted_std_image = display_depth_map(predicted_std_depth_image, min_value=0.0,
                                                                max_value=max_std_depth,
                                                                colormode=cv2.COLORMAP_HOT)
                display_predicted_std_image = cv2.cvtColor(display_predicted_std_image, cv2.COLOR_BGR2RGB)

                GIF_image_list.append(cv2.hconcat(
                    [rendering_color_image_list[i], display_simulated_depth_image,
                     display_predicted_depth_image, display_predicted_std_image]))

                if args.visualize_fused_model:
                    cv2.imshow("video_rendering_depth", cv2.cvtColor(GIF_image_list[i], cv2.COLOR_RGB2BGR))
                    cv2.waitKey(10)

            # print("Writing fly-through video of fused mesh...")
            # result_video_fp = cv2.VideoWriter(
            #     str(folder / "fused_mesh.avi"),
            #     cv2.VideoWriter_fourcc(*'DIVX'), 20,
            #     (GIF_image_list[0].shape[1], GIF_image_list[0].shape[0]))
            # for i in range(len(GIF_image_list)):
            #     result_video_fp.write(cv2.cvtColor(GIF_image_list[i], cv2.COLOR_RGB2BGR))
            # result_video_fp.release()
            # if args.visualize_fused_model:
            #     cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
