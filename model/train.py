import pickle
import time

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

from utils import AverageMeter
from tensorboard_logger import log_value


def reset(hp):
    """
    Initialize the hidden state of the core network
    and the location vector.

    This is called once every time a new minibatch
    `x` is introduced.
    """
    dtype = torch.cuda.FloatTensor

    h_t = torch.zeros(hp['BATCH_SIZE'], hp['HIDDEN_SIZE'])
    h_t = Variable(h_t).type(dtype)

    l_t = torch.Tensor(hp['BATCH_SIZE'], 2).uniform_(-1, 1)
    l_t = Variable(l_t).type(dtype)

    return h_t, l_t


def train_one_epoch(model, optimizer, train_loader, epoch, hp):
    """
    Train the model for 1 epoch of the training set.
    An epoch corresponds to one full pass through the entire
    training set in successive mini-batches.
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()

    tic = time.time()
    with tqdm(total=hp['NUM_TRAIN']) as pbar:
        for sample_index, (x, y) in enumerate(train_loader):
            x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)

            # initialize location vector and hidden state
            hp['BATCH_SIZE'] = x.shape[0]  # We need to set this to train on variable batch size
            h_t, l_t = reset(hp)

            # save images
            imgs = [x[0:9]]

            # extract the glimpses
            locs = []
            log_pi = []
            baselines = []
            for t in range(hp['NUM_GLIMPSES'] - 1):
                # forward pass through model
                h_t, l_t, b_t, p = model(x, l_t, h_t)

                # store
                locs.append(l_t[0:9])
                baselines.append(b_t)
                log_pi.append(p)

            # last iteration
            h_t, l_t, b_t, log_probas, p = model(
                x, l_t, h_t, last=True
            )
            log_pi.append(p)
            baselines.append(b_t)
            locs.append(l_t[0:9])

            # convert list to tensors and reshape
            baselines = torch.stack(baselines).transpose(1, 0)
            log_pi = torch.stack(log_pi).transpose(1, 0)

            # calculate reward
            predicted = torch.max(log_probas, 1)[1]
            R = (predicted.detach() == y).float()
            R = R.unsqueeze(1).repeat(1, hp['NUM_GLIMPSES'])

            # compute losses for differentiable modules
            loss_action = F.nll_loss(log_probas, y)
            loss_baseline = F.mse_loss(baselines, R)

            # compute reinforce loss
            # summed over timesteps and averaged across batch
            adjusted_reward = R - baselines.detach()
            loss_reinforce = torch.sum(-log_pi*adjusted_reward, dim=1)
            loss_reinforce = torch.mean(loss_reinforce, dim=0)

            # sum up into a hybrid loss
            loss = loss_action + loss_baseline + loss_reinforce

            # compute accuracy
            correct = (predicted == y).float()
            acc = 100 * (correct.sum() / len(y))

            # store
            losses.update(loss.data[0], x.size()[0])
            accs.update(acc.data[0], x.size()[0])

            # compute gradients and update SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            toc = time.time()
            batch_time.update(toc-tic)

            pbar.set_description(
                (
                    "{:.1f}s - loss: {:.3f} - acc: {:.3f}".format(
                        (toc-tic), loss.data[0], acc.data[0]
                    )
                )
            )
            pbar.update(hp['BATCH_SIZE'])

            # dump the glimpses and locs
            plot = False
            if (epoch % hp['PLOT_FREQ'] == 0) and (sample_index == 0):
                plot = True

            if plot:
                imgs = [g.cpu().data.numpy().squeeze() for g in imgs]
                locs = [l.cpu().data.numpy() for l in locs]
                pickle.dump(
                    imgs, open(
                        hp['PLOT_DIR'] + "g_{}.p".format(epoch+1),
                        "wb"
                    )
                )
                pickle.dump(
                    locs, open(
                        hp['PLOT_DIR'] + "l_{}.p".format(epoch+1),
                        "wb"
                    )
                )

            # log to tensorboard
            iteration = epoch*len(train_loader) + sample_index
            log_value('train_loss', losses.avg, iteration)
            log_value('train_acc', accs.avg, iteration)

        return losses.avg, accs.avg


def validate(model, valid_loader, epoch, hp):
    """
    Evaluate the model on the validation set.
    """
    losses = AverageMeter()
    accs = AverageMeter()

    for i, (x, y) in enumerate(valid_loader):
        x, y = x.cuda(), y.cuda()
        x, y = Variable(x), Variable(y)

        # duplicate 10 times
        x = x.repeat(hp['M'], 1, 1, 1)

        # initialize location vector and hidden state
        hp['VALIDATE_BATCH_SIZE'] = x.shape[0]
        h_t, l_t = reset(hp)

        # extract the glimpses
        log_pi = []
        baselines = []
        for t in range(hp['NUM_GLIMPSES'] - 1):
            # forward pass through model
            h_t, l_t, b_t, p = model(x, l_t, h_t)

            # store
            baselines.append(b_t)
            log_pi.append(p)

        # last iteration
        h_t, l_t, b_t, log_probas, p = model(x, l_t, h_t, last=True)
        log_pi.append(p)
        baselines.append(b_t)

        # convert list to tensors and reshape
        baselines = torch.stack(baselines).transpose(1, 0)
        log_pi = torch.stack(log_pi).transpose(1, 0)

        # average
        log_probas = log_probas.view(
            hp['M'], -1, log_probas.shape[-1]
        )
        log_probas = torch.mean(log_probas, dim=0)

        baselines = baselines.contiguous().view(
            hp['M'], -1, baselines.shape[-1]
        )
        baselines = torch.mean(baselines, dim=0)

        log_pi = log_pi.contiguous().view(
            hp['M'], -1, log_pi.shape[-1]
        )
        log_pi = torch.mean(log_pi, dim=0)

        # calculate reward
        predicted = torch.max(log_probas, 1)[1]
        R = (predicted.detach() == y).float()
        R = R.unsqueeze(1).repeat(1, hp['NUM_GLIMPSES'])

        # compute losses for differentiable modules
        loss_action = F.nll_loss(log_probas, y)
        loss_baseline = F.mse_loss(baselines, R)

        # compute reinforce loss
        adjusted_reward = R - baselines.detach()
        loss_reinforce = torch.sum(-log_pi*adjusted_reward, dim=1)
        loss_reinforce = torch.mean(loss_reinforce, dim=0)

        # sum up into a hybrid loss
        loss = loss_action + loss_baseline + loss_reinforce

        # compute accuracy
        correct = (predicted == y).float()
        acc = 100 * (correct.sum() / len(y))

        # store
        losses.update(loss.data[0], x.size()[0])
        accs.update(acc.data[0], x.size()[0])

        # log to tensorboard
        iteration = epoch*len(valid_loader) + i
        log_value('valid_loss', losses.avg, iteration)
        log_value('valid_acc', accs.avg, iteration)

    return losses.avg, accs.avg
