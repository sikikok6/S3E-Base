# Author: Jacek Komorowski
# Warsaw University of Technology

# Train on Oxford dataset (from PointNetVLAD paper) using BatchHard hard negative mining.

import os
from datetime import datetime
import numpy as np
import torch
import pickle
import tqdm
import pathlib

from torch.utils.tensorboard import SummaryWriter

from eval.evaluate import evaluate, print_eval_stats
from misc.utils import MinkLocParams, get_datetime
from models.loss import make_loss
from models.model_factory import model_factory
from models.minkloc_multimodal import MinkLocMultimodal

import torch.nn as nn


VERBOSE = False


def print_stats(stats, phase):
    if 'num_triplets' in stats:
        # For triplet loss
        s = '{} - Loss (mean/total): {:.4f} / {:.4f}    Avg. embedding norm: {:.4f}   Triplets per batch (all/non-zero): {:.1f}/{:.1f}'
        print(s.format(phase, stats['loss'], stats['total_loss'], stats['avg_embedding_norm'], stats['num_triplets'],
                       stats['num_non_zero_triplets']))
    elif 'num_pos' in stats:
        s = '{} - Mean loss: {:.6f}    Avg. embedding norm: {:.4f}   #positives/negatives: {:.1f}/{:.1f}'
        print(s.format(
            phase, stats['loss'], stats['avg_embedding_norm'], stats['num_pos'], stats['num_neg']))

    s = ''
    l = []
    if 'mean_pos_pair_dist' in stats:
        s += 'Pos dist (min/mean/max): {:.4f}/{:.4f}/{:.4f}   Neg dist (min/mean/max): {:.4f}/{:.4f}/{:.4f}'
        l += [stats['min_pos_pair_dist'], stats['mean_pos_pair_dist'], stats['max_pos_pair_dist'],
              stats['min_neg_pair_dist'], stats['mean_neg_pair_dist'], stats['max_neg_pair_dist']]
    if 'pos_loss' in stats:
        if len(s) > 0:
            s += '   '
        s += 'Pos loss: {:.4f}  Neg loss: {:.4f}'
        l += [stats['pos_loss'], stats['neg_loss']]
    if len(l) > 0:
        print(s.format(*l))

    if 'final_loss' in stats:
        # Multi loss
        # print(f"This is the print!!")
        s1 = '{} - Loss (total/final'.format(phase)
        s2 = '{:.4f} / {:.4f}'.format(stats['loss'], stats['final_loss'])
        s3 = 'Active triplets (final '
        s4 = '{:.1f}'.format(stats['final_num_non_zero_triplets'])
        if 'cloud_loss' in stats:
            s1 += '/cloud'
            s2 += '/ {:.4f}'.format(stats['cloud_loss'])
            s3 += '/cloud'
            s4 += '/ {:.1f}'.format(stats['cloud_num_non_zero_triplets'],)
        if 'image_loss' in stats:
            s1 += '/image'
            s2 += '/ {:.4f}'.format(stats['image_loss'])
            s3 += '/image'
            s4 += '/ {:.1f}'.format(stats['image_num_non_zero_triplets'],)

        s1 += '): '
        s3 += '): '
        print(s1 + s2)
        print(s3 + s4)


def tensors_to_numbers(stats):
    stats = {e: stats[e].item() if torch.is_tensor(stats[e])
             else stats[e] for e in stats}
    return stats


def do_train(dataloaders, params: MinkLocParams, debug=False):

    # Create model class
    s = get_datetime()
    model = model_factory(params)
    # model = nn.DataParallel(model)
    model_name = 'model_' + params.model_params.model + '_' + s
    print('Model name: {}'.format(model_name))
    weights_path = create_weights_folder()
    model_pathname = os.path.join(weights_path, model_name)
    if hasattr(model, 'print_info'):
        model.print_info()
    else:
        n_params = sum([param.nelement() for param in model.parameters()])
        print('Number of model parameters: {}'.format(n_params))

    # Move the model to the proper device before configuring the optimizer
    if torch.cuda.is_available():
        device = "cuda"
        model.to(device)
    else:
        device = "cpu"

    print('Model device: {}'.format(device))

    loss_fn = make_loss(params)

    params_l = []
    if isinstance(model, MinkLocMultimodal):
        # Different LR for image feature extractor (pretrained ResNet)
        if model.image_fe is not None:
            params_l.append(
                {'params': model.image_fe.parameters(), 'lr': params.image_lr})
        if model.cloud_fe is not None:
            params_l.append(
                {'params': model.cloud_fe.parameters(), 'lr': params.lr})
        if model.final_block is not None:
            params_l.append(
                {'params': model.final_net.parameters(), 'lr': params.lr})
    else:
        # All parameters use the same lr
        params_l.append({'params': model.parameters(), 'lr': params.lr})

    # Training elements
    if params.optimizer == 'Adam':
        if params.weight_decay is None or params.weight_decay == 0:
            optimizer = torch.optim.Adam(params_l)
        else:
            optimizer = torch.optim.Adam(
                params_l, weight_decay=params.weight_decay)
    elif params.optimizer == 'SGD':
        # SGD with momentum (default momentum = 0.9)
        if params.weight_decay is None or params.weight_decay == 0:
            optimizer = torch.optim.SGD(params_l)
        else:
            optimizer = torch.optim.SGD(
                params_l, weight_decay=params.weight_decay)
    else:
        raise NotImplementedError(
            'Unsupported optimizer: {}'.format(params.optimizer))
    if params.scheduler is None:
        scheduler = None
    else:
        if params.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.epochs+1,
                                                                   eta_min=params.min_lr)
        elif params.scheduler == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, params.scheduler_milestones, gamma=0.1)
        else:
            raise NotImplementedError(
                'Unsupported LR scheduler: {}'.format(params.scheduler))

    ###########################################################################
    # Initialize TensorBoard writer
    ###########################################################################

    now = datetime.now()
    logdir = os.path.join("../tf_logs", now.strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(logdir)

    ###########################################################################
    #
    ###########################################################################

    is_validation_set = 'val' in dataloaders
    if is_validation_set:
        phases = ['train', 'val']
    else:
        phases = ['train']

    # Training statistics
    stats = {'train': [], 'val': [], 'eval': []}
    current_best_recall = {'ave_one_percent_recall': 0, 'ave_recall': 0}
    for epoch in tqdm.tqdm(range(1, params.epochs + 1)):
        print(f"epoch: {epoch}")
        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_stats = []  # running stats for the current epoch

            count_batches = 0
            for batch in dataloaders[phase]:
                # batch is (batch_size, n_points, 3) tensor
                # labels is list with indexes of elements forming a batch
                count_batches += 1
                batch_stats = {}

                if debug and count_batches > 2:
                    break

                batch = {e: batch[e].to(device) for e in batch}
                # batch = batch.to(device)

                positives_mask = batch['positives_mask']
                negatives_mask = batch['negatives_mask']
                n_positives = torch.sum(positives_mask).item()
                n_negatives = torch.sum(negatives_mask).item()
                if n_positives == 0 or n_negatives == 0:
                    # Skip a batch without positives or negatives
                    print(
                        'WARNING: Skipping batch without positive or negative examples')
                    continue

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Compute embeddings of all elements
                    embeddings = model(batch)
                    # print(f"embeddings: {embeddings.shape}, pos_mask: {positives_mask.shape}, neg_mask: {negatives_mask.shape}")
                    loss, temp_stats, _ = loss_fn(
                        embeddings, positives_mask, negatives_mask)
                    # print(f"loss: {loss}")
                    # print(f"temp_stats01: {temp_stats}")
                    temp_stats = tensors_to_numbers(temp_stats)
                    # print(f"temp_stats:02 {temp_stats}")
                    batch_stats.update(temp_stats)
                    batch_stats['loss'] = loss.item()

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_stats.append(batch_stats)
                torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

            # ******* PHASE END *******
            # Compute mean stats for the phase
            epoch_stats = {}
            for key in running_stats[0].keys():
                temp = [e[key] for e in running_stats]
                epoch_stats[key] = np.mean(temp)

            stats[phase].append(epoch_stats)
            print_stats(epoch_stats, phase)

        # Evaluate the current model
        # model.eval()
        print('Evaluating the current model...')
        tmp_eval_stats = evaluate(model, device, params, silent=False)
        cur_datasets_name = "fire"
        cur_key = f"pickle/{cur_datasets_name}"
        # print(f'tmp_eval_stats: {tmp_eval_stats}')
        print('Current model results:')
        print_eval_stats(tmp_eval_stats)
        if tmp_eval_stats[cur_key]['ave_one_percent_recall'] >= current_best_recall['ave_one_percent_recall']:
            current_best_recall['ave_one_percent_recall'] = tmp_eval_stats[cur_key]['ave_one_percent_recall']
            # current_best_recall['ave_recall'] = tmp_eval_stats['pickle/fire']['ave_recall']
            _cur_avg_recall = current_best_recall['ave_one_percent_recall']
            epoch_model_path = model_pathname + \
                str(epoch) + \
                f'_epoch_current_avg_{_cur_avg_recall}_{cur_datasets_name}.pth'
            print(f"epoch_model_path: {epoch_model_path}")
            torch.save(model.state_dict(), epoch_model_path)
        if tmp_eval_stats[cur_key]['ave_recall'][0] >= current_best_recall['ave_recall']:
            # current_best_recall['ave_one_percent_recall'] = tmp_eval_stats['pickle/fire']['ave_one_percent_recall']
            current_best_recall['ave_recall'] = tmp_eval_stats[cur_key]['ave_recall'][0]
            cur_top25 = tmp_eval_stats[cur_key]['ave_recall'].tolist()
            with open(f"model_pathname_cur_{cur_datasets_name}_top25.txt", "w") as f:
                line = ""
                for i in range(len(cur_top25)-1):
                    line = line + str(cur_top25[i]) + ","
                line += str(cur_top25[-1])
                # print(line)
                f.write(line)
            _cur_recall1 = current_best_recall['ave_recall']
            epoch_model_path = model_pathname + \
                str(epoch) + \
                f'_epoch_current_recall{_cur_recall1}_{cur_datasets_name}.pth'
            print(f"epoch_model_path: {epoch_model_path}")
            torch.save(model.state_dict(), epoch_model_path)
        # model.train()

        # ******* EPOCH END *******

        # epoch_model_path = model_pathname + str(epoch)+'_epoch.pth'
        # print(f"epoch_model_path: {epoch_model_path}")
        # torch.save(model.state_dict(), epoch_model_path)
        if scheduler is not None:

            scheduler.step()

        loss_metrics = {'train': stats['train'][-1]['loss']}
        if 'val' in phases:
            loss_metrics['val'] = stats['val'][-1]['loss']
        writer.add_scalars('Loss', loss_metrics, epoch)

        if 'num_triplets' in stats['train'][-1]:
            nz_metrics = {'train': stats['train'][-1]['num_non_zero_triplets']}
            if 'val' in phases:
                nz_metrics['val'] = stats['val'][-1]['num_non_zero_triplets']
            writer.add_scalars('Non-zero triplets', nz_metrics, epoch)

        elif 'num_pairs' in stats['train'][-1]:
            nz_metrics = {'train_pos': stats['train'][-1]['pos_pairs_above_threshold'],
                          'train_neg': stats['train'][-1]['neg_pairs_above_threshold']}
            if 'val' in phases:
                nz_metrics['val_pos'] = stats['val'][-1]['pos_pairs_above_threshold']
                nz_metrics['val_neg'] = stats['val'][-1]['neg_pairs_above_threshold']
            writer.add_scalars('Non-zero pairs', nz_metrics, epoch)

        if params.batch_expansion_th is not None:
            # Dynamic batch expansion of the training batch
            epoch_train_stats = stats['train'][-1]
            if 'num_non_zero_triplets' in epoch_train_stats:
                # Ratio of non-zero triplets
                rnz = epoch_train_stats['num_non_zero_triplets'] / \
                    epoch_train_stats['num_triplets']
                if rnz < params.batch_expansion_th:
                    dataloaders['train'].batch_sampler.expand_batch()
            elif 'final_num_non_zero_triplets' in epoch_train_stats:
                rnz = []
                rnz.append(
                    epoch_train_stats['final_num_non_zero_triplets'] / epoch_train_stats['final_num_triplets'])
                if 'image_num_non_zero_triplets' in epoch_train_stats:
                    rnz.append(
                        epoch_train_stats['image_num_non_zero_triplets'] / epoch_train_stats['image_num_triplets'])
                if 'cloud_num_non_zero_triplets' in epoch_train_stats:
                    rnz.append(
                        epoch_train_stats['cloud_num_non_zero_triplets'] / epoch_train_stats['cloud_num_triplets'])
                rnz = max(rnz)
                if rnz < params.batch_expansion_th:
                    dataloaders['train'].batch_sampler.expand_batch()
            else:
                print(
                    'WARNING: Batch size expansion is enabled, but the loss function is not supported')

    print('')

    # Save final model weights
    final_model_path = model_pathname + '_final.pth'
    torch.save(model.state_dict(), final_model_path)

    stats = {'train_stats': stats, 'params': params}

    # Evaluate the final model
    model.eval()
    print('Evaluating the final model...')
    final_eval_stats = evaluate(model, device, params, silent=False)
    print('Final model results:')
    print_eval_stats(final_eval_stats)
    stats['eval'] = {'final': final_eval_stats}
    print('')

    # Pickle training stats and parameters
    pickle_path = model_pathname + '_stats.pickle'
    pickle.dump(stats, open(pickle_path, "wb"))

    # Append key experimental metrics to experiment summary file
    model_params_name = os.path.split(params.model_params.model_params_path)[1]
    config_name = os.path.split(params.params_path)[1]
    _, model_name = os.path.split(model_pathname)
    prefix = "{}, {}, {}".format(model_params_name, config_name, model_name)
    export_eval_stats("experiment_results.txt", prefix, final_eval_stats)


def export_eval_stats(file_name, prefix, eval_stats):
    s = prefix
    ave_1p_recall_l = []
    ave_recall_l = []
    # Print results on the final model
    with open(file_name, "a") as f:
        # for ds in ['oxford', 'university', 'residential', 'business']:
        for ds in ['pickle/fire', 'pickle/apt', 'pickle/apt1']:
            # if ds not in eval_stats:
            #     continue
            ave_1p_recall = eval_stats[ds]['ave_one_percent_recall']
            ave_1p_recall_l.append(ave_1p_recall)
            ave_recall = eval_stats[ds]['ave_recall'][0]
            ave_recall_l.append(ave_recall)
            s += ", {:0.2f}, {:0.2f}".format(ave_1p_recall, ave_recall)

        mean_1p_recall = np.mean(ave_1p_recall_l)
        mean_recall = np.mean(ave_recall_l)
        s += ", {:0.2f}, {:0.2f}\n".format(mean_1p_recall, mean_recall)
        f.write(s)


def create_weights_folder():
    # Create a folder to save weights of trained models
    this_file_path = pathlib.Path(__file__).parent.absolute()
    temp, _ = os.path.split(this_file_path)
    weights_path = os.path.join(temp, 'weights')
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    assert os.path.exists(
        weights_path), 'Cannot create weights folder: {}'.format(weights_path)
    return weights_path
