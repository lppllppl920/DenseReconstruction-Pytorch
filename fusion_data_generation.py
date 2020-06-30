'''
Author: Xingtong Liu, Maia Stiber, Jindan Huang, Masaru Ishii, Gregory D. Hager, Russell H. Taylor, and Mathias Unberath

Copyright (C) 2020 Johns Hopkins University - All Rights Reserved
You may use, distribute and modify this code under the
terms of the GNU GENERAL PUBLIC LICENSE Version 3 license for non-commercial usage.

You should have received a copy of the GNU GENERAL PUBLIC LICENSE Version 3 license with
this file. If not, please write to: xliu89@jh.edu or unberath@jhu.edu
'''

import tqdm
import cv2
import numpy as np
from pathlib import Path
import torch
import random
import argparse
import h5py
# Local
import models
import utils
import dataset

if __name__ == '__main__':
    cv2.destroyAllWindows()
    parser = argparse.ArgumentParser(
        description='Depth fusion data preparation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image_downsampling', type=float, default=4.0,
                        help='image downsampling rate to speed up training and reduce overfitting')
    parser.add_argument('--network_downsampling', type=int, default=64,
                        help='network downsampling rate')
    parser.add_argument('--input_size', nargs='+', type=int,
                        help='input size for the network')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for testing')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for input data loader')
    parser.add_argument('--visible_interval', type=int, default=5,
                        help='range for propagating point visibility information')
    parser.add_argument('--inlier_percentage', type=float, default=0.998,
                        help='percentage of inliers of SfM point clouds (for pruning some outliers)')
    parser.add_argument('--load_intermediate_data', action='store_true', help='whether to load intermediate data')
    parser.add_argument('--display_architecture', action='store_true', help='display the network architecture')
    parser.add_argument('--trained_model_path', type=str, required=True,
                        help='path to the trained model')
    parser.add_argument('--data_root', type=str, required=True,
                        help='root storing the video and sparse reconstruction data')
    parser.add_argument('--sequence_root', type=str, default=None,
                        help='root of the sequence')
    parser.add_argument('--patient_id', nargs='+', type=int,
                        help='list patient id')
    parser.add_argument('--precompute_root', type=str, required=True,
                        help='root of the precompute data')

    args = parser.parse_args()

    height, width = args.input_size
    # Fix randomness for reproducibility
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(10085)
    np.random.seed(10085)
    random.seed(10085)

    if not Path(args.precompute_root).exists():
        Path(args.precompute_root).mkdir(parents=True)

    model = models.FCDenseNetStd(
        in_channels=3, down_blocks=(4, 4, 4, 4, 4),
        up_blocks=(4, 4, 4, 4, 4), bottleneck_layers=4,
        growth_rate=12, out_chans_first_conv=48)
    # Initialize the depth estimation network with Kaiming He initialization
    utils.init_net(model, type="kaiming", mode="fan_in", activation_mode="relu",
                   distribution="normal")
    # Multi-GPU running
    model = torch.nn.DataParallel(model)

    # Load previous depth estimation model
    if Path(args.trained_model_path).exists():
        print("Loading {:s} ...".format(str(args.trained_model_path)))
        state = torch.load(str(args.trained_model_path), encoding='latin1')
        model.load_state_dict(state["model"])
        step = state['step']
        epoch = state['epoch']
        print('Restored model, epoch {}, step {}'.format(epoch, step))
    else:
        print("No previous model detected")
        raise OSError

    # Set model to evaluation mode
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Custom layers
    depth_scaling_layer = models.TestDepthMeanStdScalingLayer()

    folder_list = list()
    for id in args.patient_id:
        folder_list = folder_list + list(Path(args.data_root).glob("{}/_start*".format(id)))
    folder_list.sort()
    print(folder_list)

    load_intermediate_data = args.load_intermediate_data
    for folder in folder_list:
        if args.sequence_root is not None:
            if str(folder) != args.sequence_root:
                continue
        print("Start gathering fusion data for {}".format(folder))

        # if (folder / "fusion_data.hdf5").exists():
        #     continue

        image_path_list = utils.get_file_names_in_sequence(folder)
        if len(image_path_list) == 0:
            print("Sequence {} does not have relevant files".format(str(folder)))
            continue

        hf = h5py.File(str(folder / "fusion_data.hdf5"), 'w')
        dataset_extrinsics = hf.create_dataset('extrinsics', (0, 4, 4),
                                               maxshape=(None, 4, 4), chunks=(4096, 4, 4),
                                               compression="gzip", compression_opts=4, dtype='float32')
        dataset_intrinsics = hf.create_dataset('intrinsics', (0, 3, 3),
                                               maxshape=(None, 3, 3), chunks=(4096, 3, 3),
                                               compression="gzip", compression_opts=4, dtype='float32')
        dataset_mean = hf.create_dataset('mean_depth', (0, height, width, 1),
                                         maxshape=(None, height, width, 1), chunks=(1, height, width, 1),
                                         compression="gzip", compression_opts=9, dtype='float32')
        dataset_std = hf.create_dataset('std_depth', (0, height, width, 1),
                                        maxshape=(None, height, width, 1), chunks=(1, height, width, 1),
                                        compression="gzip", compression_opts=9, dtype='float32')
        dataset_color = hf.create_dataset('color', (0, height, width, 3),
                                          maxshape=(None, height, width, 3), chunks=(1, height, width, 3),
                                          compression="gzip", compression_opts=9, dtype='uint8')
        dataset_mask = hf.create_dataset('mask', (0, height, width, 1),
                                         maxshape=(None, height, width, 1), chunks=(1, height, width, 1),
                                         compression="gzip", compression_opts=9, dtype='uint8')
        dataset_frame_index = hf.create_dataset('frame_index', (0, 1),
                                                maxshape=(None, 1), chunks=(40960, 1),
                                                compression="gzip", compression_opts=4, dtype='int32')

        fusion_dataset = dataset.DepthDataset(image_file_names=image_path_list,
                                              folder_list=folder_list,
                                              image_downsampling=args.image_downsampling,
                                              network_downsampling=args.network_downsampling,
                                              inlier_percentage=args.inlier_percentage,
                                              visible_interval=args.visible_interval,
                                              load_intermediate_data=args.load_intermediate_data,
                                              intermediate_data_root=Path(args.precompute_root),
                                              num_pre_workers=args.num_workers,
                                              num_iter=None,
                                              adjacent_range=None,
                                              phase="Loading")
        fusion_loader = torch.utils.data.DataLoader(dataset=fusion_dataset, batch_size=args.batch_size, shuffle=False,
                                                    num_workers=args.num_workers)
        load_intermediate_data = True

        scale_list = []
        scaled_mean_depth_map_list = []
        scaled_std_depth_map_list = []
        extrinsics_list = []
        colors_list = []
        boundaries_list = []
        intrinsics_list = []
        image_name_list = []
        folder_list = []
        # Update progress bar
        tq = tqdm.tqdm(total=len(fusion_loader) * args.batch_size)
        with torch.no_grad():
            for batch, (colors_1, sparse_depths_1, sparse_depth_masks_1, boundaries,
                        extrinsics_1, intrinsic_matrices, image_names, folders) in enumerate(fusion_loader):
                tq.update(colors_1.shape[0])

                colors_1, sparse_depths_1, sparse_depth_masks_1, boundaries, extrinsics_1, intrinsic_matrices = \
                    colors_1.cuda(), sparse_depths_1.cuda(), sparse_depth_masks_1.cuda(), boundaries.cuda(), \
                    extrinsics_1.cuda(), intrinsic_matrices.cuda()

                colors_1 = boundaries * colors_1
                predicted_mean_depth_maps_1, predicted_std_depth_maps_1 = model(colors_1)

                scaled_mean_depth_maps_1, scaled_std_depth_maps_1, scales = depth_scaling_layer(
                    [predicted_mean_depth_maps_1, predicted_std_depth_maps_1,
                     sparse_depths_1, sparse_depth_masks_1])

                scaled_mean_depth_maps_1 = boundaries * scaled_mean_depth_maps_1
                scaled_std_depth_maps_1 = boundaries * scaled_std_depth_maps_1

                scaled_mean_depth_maps_1 = scaled_mean_depth_maps_1.data.cpu().numpy()
                scaled_std_depth_maps_1 = scaled_std_depth_maps_1.data.cpu().numpy()
                extrinsics_1 = extrinsics_1.data.cpu().numpy()
                colors_1 = colors_1.data.cpu().numpy()
                boundaries = boundaries.data.cpu().numpy()
                intrinsic_matrices = intrinsic_matrices.data.cpu().numpy()
                scales = scales.data.cpu().numpy().reshape((-1,))

                for i in range(scaled_mean_depth_maps_1.shape[0]):
                    scaled_mean_depth_map_list.append(scaled_mean_depth_maps_1[i])
                    valid_indexes = np.argwhere(boundaries[i].reshape((-1,)) > 0.5)
                    depth_vector = scaled_mean_depth_maps_1[i].reshape((-1, 1))
                    scale_list.append(scales[i])
                    scaled_std_depth_map_list.append(scaled_std_depth_maps_1[i])
                    extrinsics_list.append(extrinsics_1[i])
                    colors_list.append(colors_1[i] * 0.5 + 0.5)
                    boundaries_list.append(boundaries[i])
                    intrinsics_list.append(intrinsic_matrices[i])
                    image_name_list.append(image_names[i])

            # Use scale values to remove outlier frames. Scales should change smoothly
            recent_valid_index = 0
            valid_index_list = []
            median_scale = np.median(scale_list)
            state = "searching"
            for idx in range(len(scale_list)):
                if state == "searching":
                    ratio = scale_list[idx] / median_scale
                    if ratio >= 0.5 or ratio <= 2.0:
                        state = "normal"
                        recent_valid_index = idx
                        valid_index_list.append(idx)

                elif state == "normal":
                    ratio = scale_list[idx] / scale_list[recent_valid_index]
                    if ratio < 0.3 or ratio > 3.0:
                        print("Frame: {}, abnormal scale: {}, ratio: {}".format(idx, scale_list[idx], ratio))
                        continue
                    else:
                        recent_valid_index = idx
                        valid_index_list.append(idx)

            tq.close()
            # Write data to hdf5 file for further processing
            for i in range(len(valid_index_list)):
                idx = valid_index_list[i]
                scaled_mean_depth_map = scaled_mean_depth_map_list[idx]
                dataset_mean.resize((dataset_mean.shape[0] + 1, height, width, 1))
                dataset_mean[-1, :, :, :] = scaled_mean_depth_map.reshape((height, width, 1))

                scaled_std_depth_map = scaled_std_depth_map_list[idx]
                dataset_std.resize((dataset_std.shape[0] + 1, height, width, 1))
                dataset_std[-1, :, :, :] = scaled_std_depth_map.reshape((height, width, 1))

                color = np.moveaxis(colors_list[idx], source=[0, 1, 2], destination=[2, 0, 1])
                dataset_color.resize((dataset_color.shape[0] + 1, height, width, 3))
                dataset_color[-1, :, :, :] = np.uint8(255.0 * color.reshape((height, width, 3)))

                extrinsics = extrinsics_list[idx]
                dataset_extrinsics.resize((dataset_extrinsics.shape[0] + 1, 4, 4))
                dataset_extrinsics[-1, :, :] = extrinsics.reshape((4, 4))

                dataset_frame_index.resize((dataset_frame_index.shape[0] + 1, 1))
                dataset_frame_index[-1, :] = int(image_name_list[idx])

                if i == 0:
                    mask = boundaries_list[idx]
                    dataset_mask.resize((dataset_mask.shape[0] + 1, height, width, 1))
                    dataset_mask[-1, :, :, :] = np.uint8(mask.reshape((height, width, 1)))

                    intrinsics = intrinsics_list[idx]
                    dataset_intrinsics.resize((dataset_intrinsics.shape[0] + 1, 3, 3))
                    dataset_intrinsics[-1, :, :] = intrinsics.reshape((3, 3))

            hf.close()
