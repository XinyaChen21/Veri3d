import configargparse
from munch import *
import numpy as np
from pdb import set_trace as st

class BaseOptions():
    def __init__(self):
        self.parser = configargparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # Dataset options
        dataset = self.parser.add_argument_group('dataset')
        dataset.add_argument("--dataset_path", type=str, default='./datasets/DeepFashion')
        dataset.add_argument("--random_flip", action='store_true')
        dataset.add_argument("--gaussian_weighted_sampler", action='store_true')
        dataset.add_argument("--sampler_std", type=float, default=15)

        # Experiment Options
        experiment = self.parser.add_argument_group('experiment')
        experiment.add_argument('--config', is_config_file=True, help='config file path')
        experiment.add_argument("--expname", type=str, default='debug', help='experiment name')
        experiment.add_argument("--ckpt", type=str, default='300000', help="path to the checkpoints to resume training")
        experiment.add_argument("--continue_training", action="store_true", help="continue training the model")

        # Training loop options
        training = self.parser.add_argument_group('training')
        training.add_argument("--checkpoints_dir", type=str, default='./checkpoint', help='checkpoints directory name')
        training.add_argument("--iter", type=int, default=300000, help="total number of training iterations")
        training.add_argument("--batch", type=int, default=4, help="batch sizes for each GPU. A single RTX2080 can fit batch=4, chunck=1 into memory.")
        training.add_argument("--chunk", type=int, default=4, help='number of samples within a batch to processed in parallel, decrease if running out of memory')
        training.add_argument("--val_n_sample", type=int, default=8, help="number of test samples generated during training")
        training.add_argument("--d_reg_every", type=int, default=16, help="interval for applying r1 regularization to the StyleGAN generator")
        training.add_argument("--g_reg_every", type=int, default=4, help="interval for applying path length regularization to the StyleGAN generator")
        training.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
        training.add_argument("--mixing", type=float, default=0.9, help="probability of latent code mixing")
        training.add_argument("--lr", type=float, default=0.002, help="learning rate")
        training.add_argument("--r1", type=float, default=300, help="weight of the r1 regularization")
        training.add_argument("--path_regularize", type=float, default=2, help="weight of the path length regularization")
        training.add_argument("--path_batch_shrink", type=int, default=2, help="batch size reducing factor for the path length regularization (reduce memory consumption)")
        training.add_argument("--small_aug", action='store_true')
        training.add_argument("--adjust_gamma", action='store_true', default=False)
        training.add_argument("--gamma_lb", type=float, default=20)
        training.add_argument("--glr", type=float, default=2e-5)
        training.add_argument("--dlr", type=float, default=2e-4)

        # Inference Options
        inference = self.parser.add_argument_group('inference')
        inference.add_argument("--results_dir", type=str, default='./checkpoint', help='results/evaluations directory name')
        inference.add_argument("--truncation_ratio", type=float, default=0.5, help="truncation ratio, controls the diversity vs. quality tradeoff. Higher truncation ratio would generate more diverse results")
        inference.add_argument("--identities", type=int, default=16, help="number of identities to be generated")
        inference.add_argument("--fixed_camera_angles", action="store_true", help="when true, the generator will render indentities from a fixed set of camera angles.")
        inference.add_argument("--move_camera", action='store_true')

        # Generator options
        model = self.parser.add_argument_group('model')
        model.add_argument("--size", type=int, nargs="+", default=[256, 128], help="image sizes for the model")
        model.add_argument("--style_dim", type=int, default=256, help="number of style input dimensions")
        model.add_argument("--renderer_spatial_output_dim", type=int, nargs="+", default=[128, 64], help='spatial resolution of the StyleGAN decoder inputs')
        model.add_argument("--smpl_model_folder", type=str, default="smpl_models", help='path to smpl model folder')
        model.add_argument("--smpl_gender", type=str, default="neutral")

        # Volume Renderer options
        rendering = self.parser.add_argument_group('rendering')
        # Volume representation options
        # Ray intergration options
        rendering.add_argument("--N_samples", type=int, default=24, help='number of samples per ray')
        rendering.add_argument("--perturb", type=float, default=1., help='set to 0. for no jitter, 1. for jitter')
        # Set volume renderer outputs
        rendering.add_argument("--white_bg", action='store_true')
        # inference options
        rendering.add_argument("--render_video", action='store_true')

        rendering.add_argument("--skip_dist", type=float, default=0.1)
        # rendering.add_argument("--use_triplane", action='store_true')
        rendering.add_argument("--dataset", type=str, default="DeepFashion")
        rendering.add_argument("--multiple_sample", action='store_true', help='use multiple samples for inference to reduce flickering effects')

        self.initialized = True

    def parse(self):
        self.opt = Munch()
        if not self.initialized:
            self.initialize()
        try:
            args = self.parser.parse_args()
        except: # solves argparse error in google colab
            args = self.parser.parse_args(args=[])

        for group in self.parser._action_groups[2:]:
            title = group.title
            self.opt[title] = Munch()
            for action in group._group_actions:
                dest = action.dest
                self.opt[title][dest] = args.__getattribute__(dest)

        return self.opt
