import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import ops.tests as tests
import ops.meters as meters
import ops.norm as norm


def get_optimizer(model, name, **kwargs):
    sch_kwargs = copy.deepcopy(kwargs.pop("scheduler", {}))
    if name in ["SGD", "Sgd", "sgd"]:
        optimizer = optim.SGD(model.parameters(), **kwargs)
    elif name in ["Adam", "adam"]:
        optimizer = optim.Adam(model.parameters(), **kwargs)
    else:
        raise NotImplementedError

    sch_name = sch_kwargs.pop("name")
    if sch_name in ["MultiStepLR"]:
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **sch_kwargs)
    else:
        raise NotImplementedError

    return optimizer, train_scheduler


def train(model, optimizer,
          dataset_train, dataset_val,
          train_scheduler, warmup_scheduler,
          train_args, val_args, gpu,
          writer=None,
          verbose=1):
    train_args = copy.deepcopy(train_args)
    val_args = copy.deepcopy(val_args)

    epochs = train_args.pop("epochs")
    warmup_epochs = train_args.get("warmup_epochs", 0)
    n_ff = val_args.pop("n_ff", 1)

    model = model.cuda() if gpu else model.cpu()
    warmup_time = time.time()
    for epoch in range(warmup_epochs):
        *train_metrics, = train_epoch(optimizer, model, dataset_train, warmup_scheduler, gpu=gpu)
    if warmup_epochs > 0:
        print("The model is warmed up: %.2f sec" % (time.time() - warmup_time))

    for epoch in range(epochs):
        batch_time = time.time()
        *train_metrics, = train_epoch(optimizer, model, dataset_train,
                                             gpu=gpu)
        train_scheduler.step()
        batch_time = time.time() - batch_time

        if writer is not None and (epoch + 1) % 1 == 0:
            add_train_metrics(writer, train_metrics, epoch)
            template = "(%.2f sec/epoch) Epoch: %d, Loss: %.4f, lr: %.3e"
            print(template % (batch_time,
                              epoch,
                              train_metrics[0],
                              [param_group["lr"] for param_group in optimizer.param_groups][0]))

        if writer is not None and (epoch + 1) % 1 == 0:
            *test_metrics, cal_diag = tests.test(model, n_ff, dataset_val, verbose=False, gpu=gpu)
            add_test_metrics(writer, test_metrics, epoch)

            cal_diag = torchvision.utils.make_grid(cal_diag)
            writer.add_image("test/calibration diagrams", cal_diag, global_step=epoch)

            if verbose > 1:
                for name, param in model.named_parameters():
                    name = name.split(".")
                    writer.add_histogram("%s/%s" % (name[0], ".".join(name[1:])), param, global_step=epoch)


def train_epoch(optimizer, model, dataset,
                scheduler=None, gpu=True):
    model.train()
    nll_function = nn.CrossEntropyLoss()
    nll_function = nll_function.cuda() if gpu else nll_function

    loss_meter = meters.AverageMeter("loss")
    nll_meter = meters.AverageMeter("nll")
    l1_meter = meters.AverageMeter("l1")
    l2_meter = meters.AverageMeter("l2")

    for step, (xs, ys) in enumerate(dataset):
        if gpu:
            xs = xs.cuda()
            ys = ys.cuda()

        optimizer.zero_grad()
        logits = model(xs)
        loss = nll_function(logits, ys)
        nll_meter.update(loss.item())

        loss_meter.update(loss.item())
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

    l1_reg = norm.l1(model, gpu)
    l1_meter.update(l1_reg.item())

    l2_reg = norm.l2(model, gpu)
    l2_meter.update(l2_reg.item())

    return loss_meter.avg, nll_meter.avg, l1_meter.avg, l2_meter.avg


def add_train_metrics(writer, metrics, epoch):
    loss, nll, l1, l2 = metrics

    writer.add_scalar("train/loss", loss, global_step=epoch)
    writer.add_scalar("train/nll", nll, global_step=epoch)
    writer.add_scalar("train/l1", l1, global_step=epoch)
    writer.add_scalar("train/l2", l2, global_step=epoch)


def add_test_metrics(writer, metrics, epoch):
    nll_value, \
    cutoffs, cms, accs, uncs, ious, freqs, \
    topk_value, brier_value, \
    count_bin, acc_bin, conf_bin, ece_value, ecse_value = metrics

    writer.add_scalar("test/nll", nll_value, global_step=epoch)
    writer.add_scalar("test/acc", accs[0], global_step=epoch)
    writer.add_scalar("test/acc-90", accs[1], global_step=epoch)
    writer.add_scalar("test/unc-90", uncs[1], global_step=epoch)
    writer.add_scalar("test/iou", ious[0], global_step=epoch)
    writer.add_scalar("test/iou-90", ious[1], global_step=epoch)
    writer.add_scalar("test/freq-90", freqs[1], global_step=epoch)
    writer.add_scalar("test/top-5", topk_value, global_step=epoch)
    writer.add_scalar("test/brier", brier_value, global_step=epoch)
    writer.add_scalar("test/ece", ece_value, global_step=epoch)
