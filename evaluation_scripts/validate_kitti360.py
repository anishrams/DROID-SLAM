import sys
sys.path.append('droid_slam')
sys.path.append('thirdparty/tartanair_tools')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob
import time
import yaml
import argparse
import json

from droid import Droid
from scipy.spatial.transform import Rotation

NUM_IMAGES = 2000

def image_stream(datapath, image_size=[384, 512], intrinsics_vec=[320.0, 320.0, 320.0, 240.0], stereo=False):
    """ image generator """

    # read all png images in folder
    ht0, wd0 = [480, 640]
    images_left = sorted(glob.glob(os.path.join(datapath, 'data_rect/*.png')))[:NUM_IMAGES]
    # images_right = sorted(glob.glob(os.path.join(datapath, 'image_rcam_front/*.png')))
    images_right = None

    print("Datapath: ", datapath)
    data = []
    for t in range(len(images_left)):
        images = [ cv2.resize(cv2.imread(images_left[t]), (image_size[1], image_size[0])) ]
        if stereo:
            images += [ cv2.resize(cv2.imread(images_right[t]), (image_size[1], image_size[0])) ]

        images = torch.from_numpy(np.stack(images, 0)).permute(0,3,1,2)
        intrinsics = .8 * torch.as_tensor(intrinsics_vec)

        data.append((t, images, intrinsics))

    print("Loaded {} images".format(len(data)))

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", default="datasets/Kitti360")
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=1000)
    parser.add_argument("--image_size", default=[384,512])
    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--plot_curve", action="store_true")
    parser.add_argument("--id", type=int, default=-1)

    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=2.4)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--keyframe_thresh", type=float, default=3.5)
    parser.add_argument("--frontend_thresh", type=float, default=15)
    parser.add_argument("--frontend_window", type=int, default=20)
    parser.add_argument("--frontend_radius", type=int, default=1)
    parser.add_argument("--frontend_nms", type=int, default=1)

    parser.add_argument("--backend_thresh", type=float, default=20.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)

    args = parser.parse_args()
    torch.multiprocessing.set_start_method('spawn')

    from data_readers.kitti360 import test_split
    from evaluation.tartanair_evaluator import TartanAirEvaluator

    if not os.path.isdir("figures"):
        os.mkdir("figures")

    if args.id >= 0:
        test_split = [ test_split[args.id] ]

    ate_list = []
    all_results = {}
    for scene in test_split:
        print("Performing evaluation on {}".format(scene))
        torch.cuda.empty_cache()
        droid = Droid(args)

        scenedir = os.path.join(args.datapath, scene)
        
        for (tstamp, image, intrinsics) in tqdm(image_stream(scenedir, stereo=args.stereo)):
            droid.track(tstamp, image, intrinsics=intrinsics)

        # fill in non-keyframe poses + global BA
        traj_est = droid.terminate(image_stream(scenedir))

        ### do evaluation ###
        evaluator = TartanAirEvaluator()
        gt_file = os.path.join('data',scenedir.split('/')[-2], "cam0_to_world.txt")
        # traj_ref = np.loadtxt(gt_file, delimiter=' ')[:, [1, 2, 0, 4, 5, 3, 6]] # ned -> xyz
        gt_file_data = np.loadtxt(gt_file)
        gt_idxs = gt_file_data[:,0].astype(int)
        traj_ref = gt_file_data[gt_file_data[:,0].astype(int) < NUM_IMAGES][:,1:]
        
        # linear interpolation to match the timestamps
        li_traj_ref = []
        for i in range(1,NUM_IMAGES+1):
            if i in gt_idxs:
                li_traj_ref.append(gt_file_data[gt_file_data[:,0].astype(int) == i][:,1:].flatten())
            else:
                li_traj_ref.append(li_traj_ref[-1])
        traj_ref = np.array(li_traj_ref)
        traj_ref = traj_ref.reshape(-1,4,4) 
        xyz = traj_ref[:,:3,3]
        quat = Rotation.from_matrix(traj_ref[:,:3,:3]).as_quat()
        traj_ref = np.concatenate([xyz, quat], axis=1)

        # usually stereo should not be scale corrected, but we are comparing monocular and stereo here
        results = evaluator.evaluate_one_trajectory(
            traj_ref, traj_est, scale=True, title=scenedir[-20:].replace('/', '_'))
        
        ate_list.append(results["ate_score"])
        all_results[scene] = results

        # append the result to the result.json file
        with open('kitti360_result.json', 'w') as f:
            json.dump(all_results, f)

        traj_results = {
            "est": traj_est.tolist(),
            "ref": traj_ref.tolist(),
        }
        with open('trajectories/kitti360/{}.json'.format(scene.replace('/', '_')), 'w') as f:
            json.dump(traj_results, f)

    print("Results")
    print("ATE: ", ate_list)

    if args.plot_curve:
        import matplotlib.pyplot as plt
        ate = np.array(ate_list)
        xs = np.linspace(0.0, 1.0, 512)
        ys = [np.count_nonzero(ate < t) / ate.shape[0] for t in xs]

        plt.plot(xs, ys)
        plt.xlabel("ATE [m]")
        plt.ylabel("% runs")
        # plt.show()
        plt.savefig("figures/ate_curve_kitti360.png")

