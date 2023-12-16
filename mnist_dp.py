#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Runs MNIST training with differential privacy.

Modified from the original MNIST example mnist.py

"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from torchvision import datasets, transforms
import xgboost as xgb
import copy

# Precomputed characteristics of the MNIST dataset
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

class MLNet(nn.Module):
    def __init__(self, inputd, outputd, intersize, n_interlayers):
        super().__init__()
        self.netin = nn.Linear(inputd, intersize)
        inters = []
        for i in range(n_interlayers):
            inters.append(nn.Linear(intersize, intersize))
            inters.append(nn.ReLU())
        self.inters = torch.nn.Sequential(*inters)
        self.netout = nn.Linear(intersize, outputd)

    def forward(self, x):
        x = F.relu(self.netin(x))
        x = self.inters(x)
        y = F.relu(self.netout(x))
        return y

    def name(slef):
        return "simple mlp"



class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x

    def name(self):
        return "SampleConvNet"


class custom_bw(torch.nn.Module):
    def __init__(self):
        # A customer transform covert grayscale image tensor to a -1 and 1 matrix
        # for mnist whose shape is [1, 28, 28]
        # based on https://stackoverflow.com/questions/68415926/how-to-make-a-custom-torchvision-transform
        super().__init__()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        out = (img > 0).float() * 2.0 -1.0
        #self.out = img
        return out # Not sure how to make random_t in here

    def __repr__(self) -> str:
        return "just -1 1 tensor"

class AppendOne(torch.utils.data.Dataset):
    def __init__(self, inp_list, newmat, newlabel):
        # Just add one more item in the dataset
        # Just because opacus lib use data loader
        assert newmat.shape == inp_list[0][0].shape
        self.data_list = []
        for i in range(len(inp_list)):
            self.data_list.append((inp_list[i][0], inp_list[i][1]))
        self.data_list.append((newmat, newlabel))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx][0], self.data_list[idx][1]

class NumpyToTorchDataset(torch.utils.data.Dataset):
    def __init__(self, xtrain, ytrain):
        """
        From xgboost api's data to torch
        """

        self.data_list = []
        for x, y in zip(xtrain, ytrain):
            self.data_list.append((torch.from_numpy(x).float(), torch.from_numpy(y).float()))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx][0], self.data_list[idx][1]

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, inp_list, startind, endind):
        # left inclu  right not inclu
        # Choose a subset from dataset inp_list
        # Just because opacus lib use data loader
        assert (endind - startind) < len(inp_list)
        self.data_list = []
        for i in range(startind, endind):
            self.data_list.append((inp_list[i][0], inp_list[i][1]))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx][0], self.data_list[idx][1]


def get_known_label_sample(known_dataset):
    labels = []
    for _, lb in known_dataset:
        labels.append(lb)

    from collections import Counter
    cnter = Counter(labels)

    labprobs = torch.zeros(10)
    for i in range(10):
        if i not in cnter:
            continue
        labprobs[i] = cnter[i]

    labsample = torch.distributions.categorical.Categorical(labprobs.clone().detach())
    return labsample

def get_shadow_data(args, sample_net, known_dataset):
    labsample = get_known_label_sample(known_dataset)

    dataishape = known_dataset[0][0].shape
    ndim = np.prod(dataishape)
    mat_list = []
    pred_list = []
    shadow_models = []
    shadow_preds = []

    for i in range(args.num_guess):
        mat = 2 * ((torch.rand(ndim) -0.5) > 0).float() - 1.0
        # Random -1, 1
        mat_list.append(mat.reshape(dataishape))
        pred_list.append(labsample.sample().item())

        shadow_dataset = AppendOne(known_dataset, mat_list[-1], pred_list[-1])
        train_shadow_loader = torch.utils.data.DataLoader(
            shadow_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=True,
        )
        if args.batch_size == 1:
            model, _ = train_wodploader(args, args.shadow_epochs, train_shadow_loader, args.disable_dp)
            # Get model parameters
        else:
            model, _ = train_mnist(args, args.shadow_epochs, train_shadow_loader, args.disable_dp)

        modeldata = np.array([])
        for name, param in sample_net.state_dict().items():
            modeldata = np.concatenate([modeldata, np.copy(param.detach().numpy()).flatten()])
        shadow_models.append(modeldata)
        shadow_preds.append(mat.detach().numpy().flatten())

    shadow_models = np.array(shadow_models)
    shadow_predicts = np.array(shadow_preds)

    return shadow_models, shadow_predicts

def train_recon_tree(args, shadow_models, shadow_predicts):
    reg = xgb.XGBRegressor(
        device = args.device,
        tree_method="hist",
        n_estimators=args.num_estimators,
        learning_rate=args.tree_eta,
        reg_alpha=args.tree_alpha,
        reg_lambda=args.tree_lambda
    )
    print("Train a reconstruct xgboost")
    reg.fit(shadow_models, shadow_predicts, eval_set=[(shadow_models, shadow_predicts)])
    return reg

def train_recon_net(args, shadow_models, shadow_predicts):
    device = torch.device(args.device)
    net = MLNet(shadow_models.shape[1], shadow_predicts.shape[1], args.reco_inter, args.n_reco_inter_layers)
    net = net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.L1Loss()
    trainshadow_data = NumpyToTorchDataset(shadow_models, shadow_predicts)

    losses = []
    for epoch in range(1, args.reco_epochs + 1):
        for cnt, (data, target) in enumerate(trainshadow_data):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"train reconstruct net epoch {epoch} losses {np.mean(losses)}")

    return net

def prep_recon_model(args, known_dataset):
    net = SampleConvNet()
    shadow_models, shadow_predicts = get_shadow_data(args, net, known_dataset)
    #if args.use_rec_tree:
    reg = train_recon_tree(args, shadow_models, shadow_predicts)

    reconnet = train_recon_net(args, shadow_models, shadow_predicts)
    return reg, reconnet


def eval_recon_model(args, model, device, reg, net, original_item, prefix, check_tree=True):
    model.eval()
    modeldata = np.array([])
    for name, param in model.state_dict().items():
        modeldata = np.concatenate([modeldata, np.copy(param.detach().cpu().numpy()).flatten()])

    if check_tree:
        out = reg.predict(modeldata[None])
    else:
        net.eval()
        out = net(torch.from_numpy(modeldata).float().to(device))
        out = out.detach().cpu().numpy()

    return perf_closeness(np.copy(original_item.detach().numpy()).squeeze(), out.squeeze(), prefix)
    #perf_closeness(original_item.detach().numpy().squeeze(), netout, prefix + "renet")


def perf_closeness(original_item, out, prefix):
    out = out.reshape(original_item.shape)
    msediff = ((out - original_item) ** 2).sum()
    absdiff = np.abs(out - original_item).sum()

    #print(f"Origin\n{original_item}")
    #print(f"Recon\n{out}")

    outrel = (out > 0).astype(float) * 2 - 1
    msediffrel = ((outrel- original_item) ** 2).sum()
    absdiffrel = np.abs(outrel - original_item).sum()

    print(f"{prefix}MSE diff {msediff} abs diff {absdiff}  relative MSE diff {msediffrel} relative abs diff {absdiffrel}")

    # Save numpy as image
    # Based on  https://stackoverflow.com/questions/26929052/how-to-save-an-array-as-a-grayscale-image-with-matplotlib-numpy
    from PIL import Image
    original8 = (((original_item - original_item.min()) / (original_item.max() - original_item.min())) * 255.9).astype(np.uint8)

    img = Image.fromarray(original8)
    img.save(f"{prefix}original.png")

    reconn8 = (((out - out.min()) / (out.max() - out.min())) * 255.9).astype(np.uint8)
    reimg = Image.fromarray(reconn8)
    reimg.save(f"{prefix}recon.png")
    torch.save((original_item, out), f"{prefix}rawdata.pt")
    return msediff, absdiff, msediffrel, absdiffrel

### For gradient based attack
class gradient_logger:
    def __init__(self, ispriv=False):
        self.grades = []
        self.models = []
        self.ispriv = ispriv

    def append_record(self, epoch, index, gradients):
        self.grades.append((epoch, np.copy(index), np.copy(gradients)))

    def append_model(self, epoch, modeldict):
        modeldict = copy.deepcopy(modeldict)
        if self.ispriv == True:
            ## Change key name
            ## https://stackoverflow.com/questions/12150872/change-key-in-ordereddict-without-losing-order
            for _ in range(len(modeldict)):
                k, v = modeldict.popitem(False)
                newk = ".".join(k.split(".")[1:])
                modeldict[newk] = v

        self.models.append((epoch, modeldict))


def clip(C, value: torch.Tensor) -> torch.Tensor:
    # A clip function based on paper Hayes et al.
    return value / max(1, (value.abs()/C).max())

def recon_gradient(args, logger, known_dataset, reco_label):
    # Algorithm can be found at https://github.com/JonasGeiping/invertinggradients/blob/master/inversefed/reconstruction_algorithms.py#L130
    loss_fn = nn.CrossEntropyLoss()

    datas = []
    labels = []
    for data, target in known_dataset:
        datas.append(data)
        labels.append(target)


    datas = torch.stack(datas)
    datas.requires_grad = True
    labels = torch.tensor(labels)



    grades_epochs = []
    now_epoch = logger.grades[0][0]
    grades_sum = torch.zeros(logger.grades[0][2][0].shape)
    model_list = [SampleConvNet() for _ in range(len(logger.models))]

    for i in range(len(model_list)):
        model_list[i].load_state_dict(logger.models[i][1])

    clipC = args.clip_c
    for i in range(len(logger.grades)):
        if logger.grades[i][0] != now_epoch:
            grades_epochs.append(torch.clone(grades_sum))
            grades_sum = torch.zeros(logger.grades[0][2][0].shape)
            now_epoch = logger.grades[i][0]
        grades_sum = grades_sum + logger.grades[i][2].sum(axis=0)
    grades_epochs.append(torch.clone(grades_sum))

    assert len(grades_epochs) == len(model_list)


    best_x_trial = None
    best_loss = None
    for _ in range(args.num_trials):
        x_trial = 2 * ((torch.rand(np.prod(known_dataset[0][0].shape)) -0.5) > 0).float() - 1.0
        x_trial = x_trial.reshape(known_dataset[0][0].shape)
        x_trial.requires_grad = True

        if reco_label == None:
            labsample = get_known_label_sample(known_dataset)
            reco_label = labsample.sample().unsqueeze(0)

        optimizer = torch.optim.LBFGS([x_trial], lr=args.opt_lr)

        for e in range(len(model_list)):
            lossz_ = loss_fn(model_list[e](datas), labels)
            gradz_ = torch.autograd.grad(lossz_, datas, create_graph=False)[0]

            for i in range(len(gradz_)):
                gradz_[i] = clip(clipC, gradz_[i])

            gradz_ = gradz_.sum(axis=0)

            grades_epochs[e] = grades_epochs[e] - gradz_
            model_list[e].eval()
            for param in model_list[e].parameters():
                param.requires_grad = False

        #print(x_trial)
        for iteration in range(args.opt_max_iter):
            closure = get_gradient_closure(clipC, optimizer, x_trial, reco_label, loss_fn, model_list, grades_epochs)
            rec_loss = optimizer.step(closure)
            #print(f"{iteration} L(z) in paper changing {rec_loss}")
            if best_loss == None or rec_loss < best_loss:
                best_loss = rec_loss
                best_x_trial = x_trial.detach().clone().numpy()

    return x_trial.detach().numpy()


def get_gradient_closure(clipC, optimizer, x_trial, label, loss_fn, model_list, grades_epochs):
    def closure():
        optimizer.zero_grad()
        lt = torch.zeros(1)
        for e in range(len(grades_epochs)):
            model_list[e].zero_grad()
            loss = loss_fn(model_list[e](x_trial), label)
            gradient = torch.autograd.grad(loss, x_trial, create_graph=True)[0]

            gradient = clip(clipC, gradient)
            grad_ = grades_epochs[e]

            # Equation (1) in Hayes, Mahloujifar, Balle
            lt += - (gradient * grad_).sum() + (gradient - grad_).norm(p=1)
        lt.backward()
        return lt

    return closure


def train_wodploader(args, epochs, train_loader, disable_dp, verbose=False, test_loader=None, rcotree=None, rconet=None, original_item=None, prefix=""):
    """
    Train wo dploader based on https://github.com/pytorch/opacus/blob/main/Migration_Guide.md#no-dataloader
    """
    model = SampleConvNet().to(torch.device(args.device))
    device = torch.device(args.device)
    accountant = None
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)

    if not disable_dp:
        from opacus.accountants import RDPAccountant
        accountant = RDPAccountant()


        from opacus import GradSampleModule
        model = GradSampleModule(model)

        from opacus.optimizers import DPOptimizer
        optimizer = DPOptimizer(
            optimizer=optimizer,
            noise_multiplier=1.0, # same as make_private arguments
            max_grad_norm=1.0, # same as make_private arguments
            expected_batch_size=args.batch_size # if you're averaging your gradients, you need to know the denominator
        )
        print(args.batch_size/len(train_loader))
        optimizer.attach_step_hook(
            accountant.get_optimizer_hook_fn(
            # this is an important parameter for privacy accounting. Should be equal to batch_size / len(dataset)
            sample_rate=args.batch_size/len(train_loader)
            )
        )

    results = [[] for _ in range(epochs)]
    for epoch in range(1, epochs + 1):
        info = train(args.delta, disable_dp, model, device, train_loader, optimizer, accountant, epoch, verbose)
        rep_str = f"{prefix}priv_epoch{epoch}" if not disable_dp else f"{prefix}nonpriv_epoch{epoch}"
        results[epoch - 1].append(rep_str)
        results[epoch - 1].extend(info)
        if test_loader != None:
            results[epoch - 1].append("acc")
            results[epoch - 1].append(test(model, device, test_loader))
        if rcotree != None:
            results[epoch - 1].append("recotree")
            results[epoch - 1].extend(eval_recon_model(args, model, device, rcotree, rconet, original_item, f"{rep_str}_recotree", check_tree=True))
        if rconet != None:
            results[epoch - 1].append("reconet")
            results[epoch - 1].extend(eval_recon_model(args, model, device, rcotree, rconet, original_item, f"{rep_str}_reconet", check_tree=False))

    return model, results



def train_mnist(args, epochs, train_loader, disable_dp, verbose=False, test_loader=None, rcotree=None, rconet=None, original_item=None, prefix="", logger=None, known_dataset=None, known_label=None):
    model = SampleConvNet().to(torch.device(args.device))

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)
    account = None

    if not disable_dp:
        privacy_engine = PrivacyEngine(secure_mode=False)
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=args.sigma,
            max_grad_norm=args.max_per_sample_grad_norm,
        )
        account = privacy_engine.accountant
    results = [[] for _ in range(epochs)]
    device = torch.device(args.device)
    for epoch in range(1, epochs + 1):
        info = train(args.delta, disable_dp, model, device, train_loader, optimizer, account, epoch, verbose, logger)
        rep_str = f"{prefix}priv_epoch{epoch}" if not disable_dp else f"{prefix}nonpriv_epoch{epoch}"
        results[epoch - 1].append(rep_str)
        results[epoch - 1].extend(info)
        if test_loader != None:
            results[epoch - 1].append("acc")
            results[epoch - 1].append(test(model, device, test_loader))
        if rcotree != None:
            results[epoch - 1].append("recotree")
            results[epoch - 1].extend(eval_recon_model(args, model, device, rcotree, rconet, original_item, f"{rep_str}_recotree", check_tree=True))
        if rconet != None:
            results[epoch - 1].append("reconet")
            results[epoch - 1].extend(eval_recon_model(args, model, device, rcotree, rconet, original_item, f"{rep_str}_reconet", check_tree=False))

        if epoch == epochs and logger != None and args.gradient_based_attack == True:
            out = recon_gradient(args, logger, known_dataset, None)
            results[epoch - 1].append("gradatt_idky")
            results[epoch - 1].extend(perf_closeness(np.copy(original_item.detach().numpy()).squeeze(), out.squeeze(), f"{rep_str}_gradatt_idky"))
            out = recon_gradient(args, logger, known_dataset, torch.tensor(known_label).unsqueeze(0))
            results[epoch - 1].append("gradatt_knowny")
            results[epoch - 1].extend(perf_closeness(np.copy(original_item.detach().numpy()).squeeze(), out.squeeze(), f"{rep_str}_gradatt_knowny"))

    return model, results


def train(delta, disable_dp, model, device, train_loader, optimizer, accountant, epoch, verbose, logger=None):
    """
    logger is for recording gradient for gradient-based attack
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    for _batch_idx, (data, target) in enumerate(train_loader):
        #print(_batch_idx)
        #import pdb
        #pdb.set_trace()
        data, target = data.to(device), target.to(device)
        if logger != None:
            data.requires_grad = True
        # Check https://discuss.pytorch.org/t/newbie-getting-the-gradient-with-respect-to-the-input/12709/8
        # for gradient
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        losses.append(loss.item())
        if logger != None:
            logger.append_record(epoch, np.array(_batch_idx), data.grad.detach().cpu().numpy())
        optimizer.step()

    if logger != None:
        logger.append_model(epoch, model.state_dict())

    lossmean = np.mean(losses)
    if not disable_dp:
        epsilon = accountant.get_epsilon(delta=delta)
        if verbose:
            print(
                f"\nTrain Epoch: {epoch} \t"
                f"Loss: {lossmean:.6f} "
                f"(ε = {epsilon:.2f}, δ = {delta})"
            )
            epsilon2 = accountant.get_epsilon(delta=10 * delta)
            print(
                f"\nTrain Epoch: {epoch} \t"
                f"Loss: {lossmean:.6f} "
                f"(ε = {epsilon2:.2f}, δ = {10 * delta})"
            )
        return lossmean, epsilon, delta
    else:
        if verbose:
            print(f"\nTrain Epoch: {epoch} \t Loss: {lossmean:.6f}")
        return lossmean, None, None


def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return correct / len(test_loader.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description="Opacus MNIST Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64,
        metavar="B",
        help="Batch size",
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train",
    )
    parser.add_argument(
        "--shadow_epochs",
        type=int,
        default=4,
        help="number of epochs using to sample shadow models",
    )
    parser.add_argument(
        "-r",
        "--n-runs",
        type=int,
        default=1,
        metavar="R",
        help="number of runs to average on",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        metavar="S",
        help="Noise multiplier",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=1.0,
        metavar="C",
        help="Clip per-sample gradients to this norm",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="GPU ID for this process",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="Save the trained model",
    )
    parser.add_argument(
        "--disable-dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla SGD",
    )
    parser.add_argument(
        "--secure-rng",
        action="store_true",
        default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
    )
    parser.add_argument(
        "--tree_eta",
        type=float,
        default=0.3,
        help="range is [0, 1]"
    )
    parser.add_argument(
        "--tree_alpha",
        type=float,
        default=0,
        help="L1 regularization"
    )
    parser.add_argument(
        "--tree_lambda",
        type=float,
        default=1,
        help="L2 regularization range is [0, +inf]"
    )
    parser.add_argument(
        "--num_estimators",
        type=int,
        default=20
    )
    parser.add_argument(
        "--reco_inter",
        type=int,
        default=500
    )
    parser.add_argument(
        "--n_reco_inter_layers",
        type=int,
        default=1
    )
    parser.add_argument(
        "--reco_epochs",
        type=int,
        default=20
    )
    parser.add_argument(
        "--num_guess",
        type=int,
        default=100,
        help="How many reconstructed item to be guess?"
    )
    parser.add_argument(
        "--reco_target",
        type=int,
        default=0,
        choices=range(0, 20),
        help="The target ind to be attacked"
    )
    parser.add_argument(
        "--gradient_based_attack",
        action="store_true"
    )
    parser.add_argument(
        "--opt_lr",
        type=float,
        default=0.1
    )
    parser.add_argument(
        "--opt_max_iter",
        type=int,
        default=10,
        help="For optimizer in gradient based attack"
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=100,
        help="For optimizer in gradient based attack how many recon trial"
    )
    parser.add_argument(
        "--clip_c",
        type=float,
        default=0.1,
        help="For reconstruct loss in gradient based attack"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    data_root = "./mnist"

    all_train_dataset = datasets.MNIST(
        data_root,
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                custom_bw(),
            ]
        ),
    )
    #import time
    known_dataset = CustomDataset(all_train_dataset, 20, 12819)
    # [label for (_ , label) in all_train_dataset][:20]
    # [5, 0, 4, 1, 9, 2, 1, 3, 1, 4, 3, 5, 3, 6, 1, 7, 2, 8, 6, 9]
    target_data = all_train_dataset[args.reco_target][0]
    target_label = all_train_dataset[args.reco_target][1]

    recoreg = None
    reconet = None
    nonprivlogger = gradient_logger()
    privlogger = gradient_logger(ispriv=True)
    if not args.gradient_based_attack:
        recoreg, reconet = prep_recon_model(args, known_dataset)
        nonprivlogger = None
        privlogger = None


    train_dataset = AppendOne(known_dataset, target_data, target_label)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            data_root,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                    custom_bw(),
                ]
            ),
        ),
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    run_results = []

    repro_str = (
        f"mnist_attack{args.reco_target}_label{target_label}_recon_nondp{args.disable_dp}_guess_{args.num_guess}_"
        f"gradbased{args.gradient_based_attack}_trials{args.num_trials}_{args.opt_max_iter}_{args.clip_c}_"
        f"{args.lr}_{args.sigma}_"
        f"{args.max_per_sample_grad_norm}_{args.batch_size}_{args.epochs}"
    )

    if not (args.gradient_based_attack == True and args.disable_dp == False):
        print(f"Non private version")
        for _ in range(args.n_runs):
            model, inter_result = train_mnist(args, args.epochs, train_loader, True, verbose=True, test_loader=test_loader, rcotree=recoreg, rconet=reconet, original_item=target_data, prefix=repro_str, logger=nonprivlogger, known_dataset=known_dataset, known_label=target_label)
            run_results.append(inter_result)


    if not (args.gradient_based_attack == True and args.disable_dp == True):
        print(f"Private version")
        for _ in range(args.n_runs):
            modelpriv, inter_result = train_mnist(args, args.epochs, train_loader, False, verbose=True, test_loader=test_loader, rcotree=recoreg, rconet=reconet, original_item=target_data, prefix=repro_str, logger=privlogger, known_dataset=known_dataset, known_label=target_label)
            run_results.append(inter_result)


    for cnt, table in enumerate(run_results):
        for row in table:
            for entry in row:
                print(f"{entry},", end="")
            print()

    print(repro_str)
    print(args)
    torch.save(run_results, f"run_results_{repro_str}.pt")

    if args.save_model:
        torch.save(model.state_dict(), f"mnist_cnn_nonpriv{repro_str}.pt")
        torch.save(modelpriv.state_dict(), f"mnist_cnn_priv{repro_str}.pt")
        torch.save(reconet.state_dict(), f"mnist_reconet_{repro_str}.pt")
        recoreg.save_model(f"mnist_recotree_{repro_str}.json")


if __name__ == "__main__":
    main()
