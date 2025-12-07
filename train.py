from __future__ import print_function
import argparse
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.utils.data import DataLoader
import losses
from metric import calculate_metrics
import numpy as np
from models.model import DHSnet  # final
from dataloader_ import get_sample_index, ListDataset, data_prefetcher
from utils_HSI import sample_gt, metrics, get_device, seed_worker
from datasets import get_dataset
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser(description='PyTorch DHSnet')
parser.add_argument('--save_path', type=str, default="./results/",
                    help='the path to save the model')
parser.add_argument('--data_path', type=str, default='./data/Houston/',  # Houston,HyRANK,Pavia
                    help='the path to load the data')
parser.add_argument('--log_path', type=str, default='./logs',
                    help='the path to load the data')
parser.add_argument('--source_name', type=str, default='Houston13',  # Houston13,Dioni,PaviaU
                    help='the name of the source dir')
parser.add_argument('--target_name', type=str, default='Houston18',  # Houston18,Loukia,PaviaC
                    help='the name of the test dir')
parser.add_argument('--cuda', type=int, default=0,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")

# Training options
group_train = parser.add_argument_group('Training')
group_train.add_argument('--patch_size'
                         '', type=int, default=13,
                         help="Size of the spatial neighbourhood (optional, if "
                              "absent will be set by the model)")
group_train.add_argument('--lr', type=float, default=1e-2,
                         help="Learning rate, set by the model if not specified.")
group_train.add_argument('--weight_lmmd', type=float, default=1e+0,
                         help="Regularization parameter, controlling the contribution of lmmd loss.")
group_train.add_argument('--weight_self_training', type=float, default=1e+0,
                         help="Regularization parameter, controlling the contribution of self training loss.")
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
group_train.add_argument('--batch_size', type=int, default=100,
                         help="Batch size (optional, if absent will be set by the model")  # 100
parser.add_argument('--seed', type=int, default=2025, metavar='S',
                    help='random seed ')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--l2_decay', type=float, default=1e-4,
                    help='the L2  weight decay')

parser.add_argument('--num_epoch', type=int, default=100,
                    help='the number of epoch')
parser.add_argument('--num_trials', type=int, default=10,
                    help='the number of trials')
parser.add_argument('--training_sample_number_per_class', type=float, default=200, help='training sample number')  # 200
parser.add_argument('--threshold', default=0.95, type=float,
                    help='confidence threshold')

args = parser.parse_args()
DEVICE = get_device(args.cuda)


def train(epoch, model, train_src_dataloader, train_tar_dataloader, num_epoch):
    LEARNING_RATE = args.lr / math.pow((1 + 10 * (epoch) / num_epoch), 0.75)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=args.momentum,
                              weight_decay=args.l2_decay)

    if (epoch - 1) % 10 == 0:
        print('learning rate{: .4f}'.format(LEARNING_RATE))

    global writer
    epoch_loss, CNN_cls_loss, lmmd_loss, selftraining_loss = 0, 0, 0, 0
    CNN_correct, CNN_tar_correct = 0, 0
    len_tar_temp = 0

    iter_source = data_prefetcher(train_src_dataloader)
    iter_target = data_prefetcher(train_tar_dataloader)
    num_iter = len_src_loader
    bs = train_src_dataloader.batch_size
    src_samples = np.zeros(num_classes)

    for i in range(num_iter):

        model.train()
        data_src, label_src = iter_source.next()
        data_tar, label_tar = iter_target.next()
        label_src = label_src - 1
        label_tar = label_tar - 1
        for ClassNo in range(num_classes):
            src_samples[ClassNo] += (label_src == ClassNo).sum()

        optimizer.zero_grad()
        s_pred, t_label, source_share, target_share, y_pseudo = model(data_src, data_tar)

        # loss
        loss_cls = F.nll_loss(F.log_softmax(s_pred, dim=1), label_src.long())

        lmmd = losses.LMMD_loss(class_num=num_classes)
        loss_lmmd = lmmd.get_loss(source_share, target_share, label_src.long(),
                                  torch.nn.functional.softmax(t_label, dim=1))

        self_training_criterion = losses.ConfidenceBasedSelfTrainingLoss(args.threshold)
        loss_self_training, mask, _ = self_training_criterion(y_pseudo, t_label)

        loss_total = loss_cls + args.weight_self_training * loss_self_training
        selftraining_loss += loss_self_training.item()
        lmmd_loss += loss_lmmd.item()

        loss_total.backward()
        optimizer.step()

        CNN_cls_loss += loss_cls.item()
        epoch_loss += loss_total.item()

        pred = s_pred.data.max(1)[1]
        CNN_correct += pred.eq(label_src.data.view_as(pred)).cpu().sum()
        pred_tar = t_label.data.max(1)[1]
        len_tar_temp += len(pred_tar)

        if len_tar_train_dataset - len_tar_temp >= 0:
            CNN_tar_correct += pred_tar.eq(label_tar.data.view_as(pred_tar)).cpu().sum()
            len_tar = len_tar_temp

        if i % (num_iter // 4) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(epoch + 1, i * len(data_src), len_src_dataset,
                                                             100. * i / len_src_loader))
            print(
                'loss_total: {:.5f},  loss_cls: {:.5f},  loss_lmmd:{:.4f}, loss_self_training:{:.4f}, pseudo_label_ratio:{:.0f}%'.format(
                    loss_total.item(), loss_cls.item(), loss_lmmd.item(), loss_self_training.item(),
                    mask.mean().item() * 100))

    CNN_acc = CNN_correct.item() / len_src_dataset
    CNN_tar_acc = CNN_tar_correct.item() / len_tar
    CNN_cls_loss /= len_src_loader
    lmmd_loss /= len_src_loader
    selftraining_loss /= len_src_loader
    epoch_loss /= len_src_loader


def test(model):
    model.eval()
    loss = 0
    correct = 0
    pred_list, label_list = [], []
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.cuda(), label.cuda()
            label = label - 1
            soft_pred = model.predict(data)
            pred = soft_pred.data.max(1)[1]
            pred_list.append(pred.cpu().numpy())
            label_list.append(label.cpu().numpy())
            loss += F.nll_loss(F.log_softmax(soft_pred, dim=1), label.long()).item()  # sum up batch loss
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()

        loss /= len_tar_loader
        print('Testing...')
        print('{} set: Average test loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n, | Test sample number: {:6}'.format(
            args.target_name, loss, correct, len_tar_dataset,
            100. * correct / len_tar_dataset, len_tar_dataset))
    return correct, correct.item() / len_tar_dataset, pred_list, label_list


if __name__ == '__main__':
    seed_worker(args.seed)
    num_classes = 0
    acc_test_list = np.zeros([args.num_trials, 1])

    for flag in range(args.num_trials):
        img_src, gt_src_, LABEL_VALUES_src, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.source_name,
                                                                                             args.data_path)
        img_tar, gt_tar_, LABEL_VALUES_tar, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.target_name,
                                                                                             args.data_path)

        sample_num_src = len(np.nonzero(gt_src_)[0])
        sample_num_tar = len(np.nonzero(gt_tar_)[0])

        num_classes = int(gt_tar_.max())
        N_BANDS = img_src.shape[-1]
        N_ROWS = img_tar.shape[0]
        N_COLS = img_tar.shape[1]

        hyperparams = vars(args)
        hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS,
                            'device': DEVICE, 'center_pixel': False, 'supervision': 'full'})
        hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

        r = int(hyperparams['patch_size'] / 2) + 1
        img_src = np.pad(img_src, ((r, r), (r, r), (0, 0)), 'symmetric')
        img_tar = np.pad(img_tar, ((r, r), (r, r), (0, 0)), 'symmetric')
        gt_src = np.pad(gt_src_, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
        gt_tar = np.pad(gt_tar_, ((r, r), (r, r)), 'constant', constant_values=(0, 0))

        # Generate the dataset
        train_src_index = get_sample_index(gt_src, args.training_sample_number_per_class)
        train_tar_index = get_sample_index(gt_tar, args.training_sample_number_per_class)
        test_gt_tar, _, testing_set, _ = sample_gt(gt_tar, 1, mode='random')

        g = torch.Generator()
        g.manual_seed(args.seed)

        train_dataset = ListDataset(img_src, gt_src, train_src_index, args.patch_size)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
        train_tar_dataset = ListDataset(img_tar, gt_tar, train_tar_index, args.patch_size)
        train_tar_loader = DataLoader(train_tar_dataset, shuffle=True, batch_size=args.batch_size)
        test_dataset = ListDataset(img_tar, test_gt_tar, np.argwhere(test_gt_tar != 0), args.patch_size)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)

        len_src_loader = len(train_loader)
        len_tar_train_loader = len(train_tar_loader)
        len_src_dataset = len(train_loader.dataset)
        len_tar_train_dataset = len(train_tar_loader.dataset)
        len_tar_dataset = len(test_loader.dataset)
        len_tar_loader = len(test_loader)

        print(hyperparams)
        print("train samples :", len_src_dataset)
        print("train tar samples :", len_tar_train_dataset)
        print("test samples:")
        print("Class No.      Numbers of source domain samples")

        for ClassNo in range(1, num_classes + 1):
            print("%-16s%-16d" % (ClassNo, (gt_src == ClassNo).sum()))
        print("total          ", sample_num_tar)

        model_DHSnet = DHSnet(N_BANDS, num_classes=gt_src.max().astype(int),
                              patch_size=hyperparams['patch_size']).to(DEVICE)
        total_trainable_params = sum(p.numel() for p in model_DHSnet.parameters() if p.requires_grad)
        print(f'{total_trainable_params / (1024 * 1024):.2f}M training parameters.')

        correct = 0
        timelist = []

        for epoch in range(args.num_epoch):
            t1 = time.time()
            train(epoch, model_DHSnet, train_loader, train_tar_loader, args.num_epoch)
            t2 = time.time()
            timelist.append(t2 - t1)

            if epoch % args.log_interval == 0:
                t_correct, Current_acc, pred, label = test(model_DHSnet)
                if t_correct > correct:
                    correct = t_correct
                    acc = Current_acc
                    if acc > 0.5:
                        acc_test_list[flag] = acc
                        results = metrics(np.concatenate(pred), np.concatenate(label),
                                          ignored_labels=hyperparams['ignored_labels'], n_classes=gt_src.max())
                        print(classification_report(np.concatenate(pred), np.concatenate(label),
                                                    target_names=LABEL_VALUES_tar))
                        # model_save_path = os.path.join(args.save_path, 'DHSnet_params_' + args.source_name + '_' + str(
                        #     int(acc * 100)) + '.pkl')
                        # torch.save(model_DHSnet.state_dict(), model_save_path)

            print('source: {} to target: {} max correct: {} max accuracy{: .2f}%\n'.format(
                args.source_name, args.target_name, correct, 100. * correct / len_tar_dataset))
        print(f"average train time for one epoch: {np.mean(timelist):.2f} s")

        # output classification map
        prediction_matrix = np.zeros((N_ROWS, N_COLS), dtype=int)
        for index in range(sample_num_tar):
            prediction_matrix[int(test_dataset.indices[index][0] - r)][int(test_dataset.indices[index][1] - r)] = \
                results['prediction'][index] + 1

        out = calculate_metrics(prediction_matrix, gt_tar_)
        print("result:")
        for ClassNo in range(num_classes):
            print("%-16s%.2f%%" % (ClassNo + 1, out[ClassNo] * 100))
        print("%-16s%.2f%%" % ("OA(%)", out[num_classes] * 100))
        print("%-16s%.2f%%" % ("AA(%)", out[num_classes + 1] * 100))
        print("%-16s%.2f" % ("KAPPA×100", out[num_classes + 2] * 100))




