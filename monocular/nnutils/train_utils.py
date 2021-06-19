"""
Generic Training Utils.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import os
import os.path as osp
import time
import pdb
from absl import flags

from utils.visualizer import Visualizer
from tqdm import tqdm

# -------------- flags -------------#
# ----------------------------------#
## Flags for training
curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'misc', 'cachedir')

flags.DEFINE_string('name', 'exp_name', 'Experiment Name')
# flags.DEFINE_string('cache_dir', cache_path, 'Cachedir') # Not used!
flags.DEFINE_integer('gpu_id', 0, 'Which gpu to use')
flags.DEFINE_integer('num_epochs', 1000, 'Number of epochs to train')
flags.DEFINE_integer('num_pretrain_epochs', 0, 'If >0, we will pretain from an existing saved model.')
flags.DEFINE_float('learning_rate', 0.0001, 'learning rate')
flags.DEFINE_float('beta1', 0.9, 'Momentum term of adam')

flags.DEFINE_bool('use_sgd', False, 'if true uses sgd instead of adam, beta1 is used as mmomentu')
flags.DEFINE_bool('warmup', False, 'warm up camera')
flags.DEFINE_bool('texture_warmup', False, 'warm up camera')
flags.DEFINE_bool('load_warmup', False, 'warm up camera')
flags.DEFINE_bool('drop_hypothesis', False, 'warm up camera')

flags.DEFINE_integer('batch_size', 16, 'Size of minibatches')
flags.DEFINE_integer('num_iter', 0, 'Number of training iterations. 0 -> Use epoch_iter')
flags.DEFINE_integer('num_reps', 20, 'number of repetitions')
flags.DEFINE_integer('tex_num_reps', 20, 'number of repetitions')
## Flags for logging and snapshotting
flags.DEFINE_string('checkpoint_dir', osp.join(cache_path, 'snapshots'),
                    'Root directory for output files')
# flags.DEFINE_integer('print_freq', 20, 'scalar logging frequency')
flags.DEFINE_integer('print_freq', 1, 'scalar logging frequency')
flags.DEFINE_integer('save_latest_freq', 10000, 'save latest model every x iterations')
flags.DEFINE_integer('save_epoch_freq', 50, 'save model every k epochs')

## Flags for visualization
# flags.DEFINE_integer('display_freq', 100, 'visuals logging frequency')
flags.DEFINE_integer('display_freq', 1, 'visuals logging  frequency')
flags.DEFINE_boolean('display_visuals', True, 'whether to display images')
# flags.DEFINE_boolean('display_visuals', False, 'whether to display images')
flags.DEFINE_boolean('print_scalars', True, 'whether to print scalars')
# flags.DEFINE_boolean('plot_scalars', False, 'whether to plot scalars')
flags.DEFINE_boolean('plot_scalars', True, 'whether to plot scalars')
flags.DEFINE_boolean('is_train', True, 'Are we training ?')
flags.DEFINE_integer('display_id', 1, 'Display Id')
flags.DEFINE_integer('display_winsize', 256, 'Display Size')
flags.DEFINE_integer('display_port', 8097, 'Display port')
flags.DEFINE_integer('display_single_pane_ncols', 0,
                     'if positive, display all images in a single visdom web panel with certain number of images per row.')


# -------- tranining class ---------#
# ----------------------------------#
class Trainer():
    def __init__(self, opts):
        self.opts = opts
        self.gpu_id = opts.gpu_id
        self.Tensor = torch.cuda.FloatTensor if (self.gpu_id is not None) else torch.Tensor
        self.invalid_batch = False  # the trainer can optionally reset this every iteration during set_input call
        self.save_dir = os.path.join(opts.checkpoint_dir, opts.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        log_file = os.path.join(self.save_dir, 'opts.log')
        with open(log_file, 'w') as f:
            for k in dir(opts):
                f.write('{}: {}\n'.format(k, opts.__getattr__(k)))

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_id=None):
        save_filename = '{}_net_{}.pth'.format(network_label, epoch_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if gpu_id is not None and torch.cuda.is_available():
            network.cuda(device=gpu_id)
        return

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, network_dir=None):
        save_filename = '{}_net_{}.pth'.format(network_label, epoch_label)
        if network_dir is None:
            network_dir = self.save_dir
        save_path = os.path.join(network_dir, save_filename)
        try:
            network.load_state_dict(torch.load(save_path))
        except Exception as e:
            print(e)
            network.load_state_dict(torch.load(save_path), strict=False)
        return

    def define_model(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def init_dataset(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def define_criterion(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def set_input(self, batch):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def forward(self):
        '''Should compute self.total_loss. To be implemented by the child class.'''
        raise NotImplementedError

    def save(self, epoch_prefix):
        '''Saves the model.'''
        self.save_network(self.model, 'pred', epoch_prefix, gpu_id=self.opts.gpu_id)
        return

    def get_current_visuals(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def get_current_scalars(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def get_current_points(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def init_training(self):
        opts = self.opts
        self.init_dataset()
        self.define_model()
        self.define_criterion()
        self.params_general = []
        self.params_cameras = []
        for name, param in self.model.named_parameters():
            if 'cameras' not in name:
                self.params_general.append(param)
            else:
                print(name)
                self.params_cameras.append(param)
        if opts.use_sgd:
            self.optimizer = torch.optim.SGD([
                {'params': self.params_general},
                {'params': self.model.mean_v, 'lr': 1e-1}],
                {'params': self.model.lbs, 'lr': 1e-1},
                {'params': self.model.vert2kp, 'lr': 1e-1},
                lr=opts.learning_rate, momentum=opts.beta1)
        else:

            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=opts.learning_rate, betas=(opts.beta1, 0.999))

        if opts.warmup:
            self.warmup_optimizer = torch.optim.Adam([c.weight for c in self.model.cameras], lr=1e-1)


    def train(self):
        opts = self.opts
        self.smoothed_total_loss = 0
        self.visualizer = Visualizer(opts)
        visualizer = self.visualizer
        total_steps = 0
        dataset_size = len(self.dataloader)

        if opts.warmup and not opts.load_warmup:
            for i, batch in enumerate(tqdm(self.dataloader_warmup)):

                self.set_input(batch)
                max_iters = 20
                for _ in range(max_iters):
                    if not self.invalid_batch:
                        self.warmup_optimizer.zero_grad()
                        self.warmup()
                        print(i, self.total_loss.item())
                        self.total_loss.backward()
                        self.warmup_optimizer.step()
                        if (total_steps + 1) % opts.D_mask_iters == 0:
                            if opts.mask_disc_wt > 0:
                                self.optimizerD.zero_grad()
                                self.forward_d()
                                self.loss_D_mask.backward()
                                self.optimizerD.step()
        print('end of pose warmup')
        self.save('warmup')
        total_steps = 0
        if opts.texture_warmup and not opts.load_warmup:
            for wi in tqdm(range(opts.tex_num_reps)):
                for i, batch in enumerate(self.dataloader):
                    if i > 3:
                        break
                    self.set_input(batch)
                    if not self.invalid_batch:
                        self.optimizer.zero_grad()
                        self.forward(detach_camera=False, drop_deform=True)
                        # self.forward(detach_camera=True)
                        self.smoothed_total_loss = self.smoothed_total_loss * 0.99 + 0.01 * self.total_loss.item()
                        self.total_loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()


                        if opts.display_visuals and (total_steps % opts.display_freq == 0):
                            visualizer.display_current_results(self.get_current_visuals(), wi)
                        total_steps += 1
        print('end of texture warmup')
        self.save('texture_warmup')

        for epoch in range(opts.num_pretrain_epochs, opts.num_epochs):
            epoch_iter = 0
            if epoch <= 30 and opts.drop_hypothesis:
                opts.num_guesses = opts.num_guesses
            elif epoch <= 100 and opts.drop_hypothesis:
                opts.num_guesses = min(4, opts.num_guesses)  # drop most hypothesis
            elif opts.drop_hypothesis:
                opts.num_guesses = min(4, opts.num_guesses)
            for i, batch in enumerate(self.dataloader):
                iter_start_time = time.time()
                self.set_input(batch)
                if not self.invalid_batch:
                    self.optimizer.zero_grad()
                    self.forward()
                    self.smoothed_total_loss = self.smoothed_total_loss * 0.99 + 0.01 * self.total_loss.item()
                    self.total_loss.backward()
                    self.optimizer.step()

                    total_steps += 1
                epoch_iter += 1

                if opts.display_visuals and (total_steps % opts.display_freq == 0):
                    iter_end_time = time.time()
                    print('time/itr %.2g' % ((iter_end_time - iter_start_time) / opts.display_freq))
                    visualizer.display_current_results(self.get_current_visuals(), epoch)
                    visualizer.plot_current_points(self.get_current_points())

                if opts.print_scalars and (total_steps % opts.print_freq == 0):
                    scalars = self.get_current_scalars()
                    visualizer.print_current_scalars(epoch, epoch_iter, scalars)
                    if opts.plot_scalars:
                        visualizer.plot_current_scalars(epoch, float(epoch_iter) / dataset_size, opts, scalars)

                if total_steps % opts.save_latest_freq == 0:
                    print('saving the model at the end of epoch {:d}, iters {:d}'.format(epoch, total_steps))
                    self.save('latest')

                if total_steps == opts.num_iter:
                    return

            if (epoch + 1) % opts.save_epoch_freq == 0:
                print('saving the model at the end of epoch {:d}, iters {:d}'.format(epoch, total_steps))
                self.save('latest')
                self.save(epoch + 1)
