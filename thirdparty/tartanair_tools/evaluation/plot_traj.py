import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import os
from evaluate_ate_scale import align
from transformation import pos_quats2SE_matrices, SE2pos_quat

class TrajectoryPlotter:
    def __init__(self, traj_ref, traj_est):
        self.traj_ref = traj_ref
        self.traj_est = traj_est
        self.align_trajectories()

    def plot(self, save_path=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.traj_ref[:,0], self.traj_ref[:,1], self.traj_ref[:,2], label='Ground Truth')
        ax.plot(self.traj_est[:,0], self.traj_est[:,1], self.traj_est[:,2], label='Estimated')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        if save_path:
            print('Saving plot to', save_path)
            plt.savefig(save_path)

    def align_trajectories(self):
        gt_xyz = np.matrix(self.traj_ref[:, :3].transpose())
        est_xyz = np.matrix(self.traj_est[:, :3].transpose())
        rot, trans, trans_error, s = align(gt_xyz, est_xyz,calc_scale=True)
        ate_error = np.sqrt(np.dot(trans_error,trans_error) / len(trans_error))

        print('Scale:', s)
        print('ATE:', ate_error)

        # align two trajs 
        est_SEs = pos_quats2SE_matrices(self.traj_est)
        T = np.eye(4) 
        T[:3,:3] = rot
        T[:3,3:] = trans 
        T = np.linalg.inv(T)
        est_traj_aligned = []
        for se in est_SEs:
            se[:3,3] = se[:3,3] * s
            se_new = T.dot(se)
            se_new = SE2pos_quat(se_new)
            est_traj_aligned.append(se_new)

        self.traj_est = np.array(est_traj_aligned)
        ate_error = np.sqrt((self.traj_ref[:, :3] - self.traj_est[:, :3])**2).mean()/s

        print('Aligned ATE:', ate_error)

# Example usage
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", default="datasets/Kitti360")
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--dataset", default="kitti360")

    args = parser.parse_args()

    if args.dataset == 'kitti360':
        json_files = [f for f in os.listdir('trajectories/kitti360') if f.endswith('.json')]
        for json_file in json_files:
            with open('trajectories/kitti360/' + json_file, 'r') as f:
                scene = json_file.split('.')[0]
                data = json.load(f)
                traj_ref = np.array(data['ref'])
                traj_est = np.array(data['est'])

                plotter = TrajectoryPlotter(traj_ref, traj_est)
                plotter.plot(save_path='figures/trajectory_kitti360_{}.png'.format(scene))

    elif args.dataset == 'tartanair':
        # files ending with .json
        json_files = [f for f in os.listdir('trajectories/tartanair') if f.endswith('.json')]
        for json_file in json_files:
            with open('trajectories/tartanair/' + json_file, 'r') as f:
                scene = json_file.split('.')[0]
                data = json.load(f)
                traj_ref = np.array(data['ref'])
                traj_est = np.array(data['est'])

                plotter = TrajectoryPlotter(traj_ref, traj_est)
                plotter.plot(save_path='figures/trajectory_tartanair_{}.png'.format(scene))


