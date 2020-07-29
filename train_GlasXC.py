from datetime import datetime
from functools import partial
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from operator import itemgetter
from scipy.sparse.linalg import norm
from scipy.sparse import csr_matrix, vstack

from XMC.GlasXC import GlasXC
from XMC.loaders import LibSVMLoader
from XMC.metrics import precision_at_k, ndcg_score_at_k, ps_precision_at_k
from XMC.metrics.cmetrics import evaluations, input_dropout, stocastic_negative_sampling

from xclib.data import data_utils
import xclib.evaluation.xc_metrics as xc_metrics
from MIPS import *
from sklearn.preprocessing import MultiLabelBinarizer
from evaluation import *

import random
import math
import os
import yaml
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


def csr2list(A):
    res = []
    for i in range(A.shape[0]):
        res.append(list(A[i].indices))
    return res


def weights_init(mdl, scheme):
    """
    Function to initialize weights

    Args:
        mdl : Module whose weights are going to modified
        scheme : Scheme to use for weight initialization
    """
    if isinstance(mdl, torch.nn.Linear):
        func = getattr(torch.nn.init, scheme + '_')  # without underscore is deprecated
        func(mdl.weight)


TIME_STAMP = datetime.utcnow().isoformat()

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

# data argument
parser.add_argument('--data_root', type=str, required=True,
                    help="""Root folder for dataset.
                            Note that the root folder should contain files either ending with
                            test / train""")
parser.add_argument('--dataset_info', type=str, required=True,
                    help='Dataset information in YAML format')

# architecture arguments
parser.add_argument('--input_encoder_cfg', type=str, required=True,
                    help='Input Encoder architecture configuration in YAML format')
parser.add_argument('--input_decoder_cfg', type=str, required=True,
                    help='Input Decoder architecture configuration in YAML format')
parser.add_argument('--output_encoder_cfg', type=str, required=True,
                    help='Output Encoder architecture configuration in YAML format')
parser.add_argument('--output_decoder_cfg', type=str, required=True,
                    help='Output Decoder architecture configuration in YAML format')
parser.add_argument('--regressor_cfg', type=str, required=True,
                    help='Regressor architecture configuration in YAML format')
parser.add_argument('--init_scheme', type=str, default='default',
                    choices=['xavier_uniform', 'kaiming_uniform', 'default'])

# training configuration arguments
parser.add_argument('--device', type=str, default='cpu',
                    help='PyTorch device string <device_name>:<device_id>')
parser.add_argument('--seed', type=int, default=None,
                    help='Manually set the seed for the experiments for reproducibility')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Batch size for training')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of epochs to train the autoencoder for')
parser.add_argument('--interval', type=int, default=-1,
                    help='Interval between two status updates on training')
parser.add_argument('--input_ae_loss_weight', type=float, default=1.,
                    help='Weight to give the input autoencoder loss in the entire loss')
parser.add_argument('--output_ae_loss_weight', type=float, default=1.,
                    help='Weight to give the output autoencoder loss in the entire loss')
parser.add_argument('--plot', action='store_true',
                    help='Option to plot the loss variation over iterations')

# optimizer arguments
parser.add_argument('--optimizer_cfg', type=str, required=True,
                    help='Optimizer configuration in YAML format for GlasXC model')

# post training arguments
parser.add_argument('--save_model', type=str, default=None,
                    choices=['all', 'inputAE', 'outputAE', 'regressor'], nargs='+',
                    help='Options to save the model partially or completely')
parser.add_argument('--k', type=int, default=5,
                    help='k for Precision at k and NDCG at k')
parser.add_argument('--A', type=int, default=0.55,
                    help='A for Propensity score')
parser.add_argument('--B', type=int, default=1.5,
                    help='B for Propensity score')

# parse the arguments
args = parser.parse_args()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CUDA Capability ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if args.device == 'cuda':
    print ('using gpu')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

cur_device = torch.device(args.device)
USE_CUDA = cur_device.type == 'cuda'
if USE_CUDA and not torch.cuda.is_available():
    raise ValueError("You can't use CUDA if you don't have CUDA")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Check Num of CPU's~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Num of threads : ",torch.get_num_threads())


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Reproducibility ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if args.seed is not None:
    torch.manual_seed(args.seed)
    if USE_CUDA:
        torch.cuda.manual_seed(args.seed)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Model initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

input_enc_cfg = yaml.load(open(args.input_encoder_cfg))
input_dec_cfg = yaml.load(open(args.input_decoder_cfg))
output_enc_cfg = yaml.load(open(args.output_encoder_cfg))
output_dec_cfg = yaml.load(open(args.output_decoder_cfg))
regress_cfg = yaml.load(open(args.regressor_cfg))

Glas_XC = GlasXC(input_enc_cfg, input_dec_cfg, output_enc_cfg, output_dec_cfg, regress_cfg)
if args.init_scheme != 'default':
    Glas_XC.apply(partial(weights_init, scheme=args.init_scheme))
Glas_XC = Glas_XC.to(cur_device)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Optimizer initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

opt_options = yaml.load(open(args.optimizer_cfg))
optimizer = getattr(torch.optim, opt_options['name'])(Glas_XC.parameters(),
                                                      **opt_options['args'])
#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Dataloader initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

loader_kwargs = {}

dset_opts = yaml.load(open(args.dataset_info))
USE_TEST_DSET = 'test_filename' in dset_opts.keys()

train_file = os.path.join(args.data_root, dset_opts['train_filename'])
train_loader = LibSVMLoader(train_file, dset_opts['train_opts'])
len_loader = len(train_loader)

inv_propen = xc_metrics.compute_inv_propesity(train_loader.classes, args.A, args.B)
#inv_propen = torch.from_numpy(inv_propen).to(device=cur_device)
#print (inv_propen)
label_count = np.sum(train_loader.classes, 0)
label_count = torch.from_numpy(label_count).to(cur_device)

### feature normalization
row_sums = norm(train_loader.features, axis=1)
row_indices, _ = train_loader.features.nonzero()
train_loader.features.data /= row_sums[row_indices]

train_data_loader = torch.utils.data.DataLoader(train_loader, batch_size=args.batch_size,
                                                shuffle=True, **loader_kwargs)


if USE_TEST_DSET:
    test_file = os.path.join(args.data_root, dset_opts['test_filename'])
    test_loader = LibSVMLoader(test_file, dset_opts['test_opts'])

    ### feature normalization
    row_sums = norm(test_loader.features, axis=1)
    row_indices, _ = test_loader.features.nonzero()
    test_loader.features.data /= row_sums[row_indices]

    test_data_loader = torch.utils.data.DataLoader(test_loader, batch_size=4096,
                                                   shuffle=False)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Train your model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

all_iters = 0
ALPHA_INPUT = args.input_ae_loss_weight
ALPHA_OUTPUT = args.output_ae_loss_weight
K = args.k
INP_REC_LOSS = []
OTP_REC_LOSS = []
CLASS_LOSS = []
AVG_P_AT_K = []
AVG_NDCG_AT_K = []
LAMBDA = 10 # default = 10
mean = 0.5

#epsilon = 10e-5                # Use only for Eurlex because Z becomes singular

for epoch in range(args.epochs):
    #print("In epoch ",epoch)
    cur_no = 0
    for x, y in iter(train_data_loader):

        #print (x.size(0), x.size(1))
        #~~~~~~~~apply input dropout~~~~~~~~~
        #x = torch.from_numpy(input_dropout(csr_matrix(x.cpu().numpy()), rho=0.2).toarray())

        cur_no += x.size(0)
        #~~~~~~~~~~stocastic negative sampling~~~~~~~~~~~~~~

        optimizer.zero_grad()

        x = x.to(device=cur_device, dtype=torch.float)
        y = y.to(device=cur_device, dtype=torch.float)

        F.dropout(x, p=0.2, inplace=True)

        inp_ae_fp, out_ae_fp, reg_fp, V = Glas_XC.forward(x, y)
        #print (V)
        #V = F.normalize(V, dim=0)
        #print("Size of reg_fp is : ", reg_fp.size())
        #print("Size of decoder weight is : ", V.size())
        #print("The first two columns of decoder matrix is :", V[:, :2])
        #print(type(decoder_weight_mat[1,1]))


        # Build GLAS Regularizer

        y_sum = torch.sum(y, dim=1)
        #print (y.size())
        for s in (y_sum == 0).nonzero():
            #y[s] = torch.from_numpy(np.copy(y[s-1].numpy())) # if some sample has no labels, copy the previous sample
            y[s, torch.randint(0, y.size(1), (1,))] = 1 # setting a random label as 1

        activate_labels = torch.nonzero(y, as_tuple=True)[1]
        #print (activate_labels.size())

        activate_labels = torch.unique(activate_labels)
        #sampled_labels = torch.multinomial(1 / (1 + label_count[0, activate_labels]), y.size(0))
        sampled_labels = torch.randint(0, activate_labels.size(0), (y.size(0),)).to(device=cur_device)
        #print (sampled_labels)
        #print (sampled_labels.size())
        #print (activate_labels)
        #print (activate_labels.size())
        sampled_labels = activate_labels[sampled_labels]

        y_sampled = torch.index_select(y, 1, sampled_labels)  #indexes the input tensor along column using the entries in indices

        div = 1/math.pow(sampled_labels.size(0), 2) # different for different datasets
        #print("Size of sampled y is : ", y_sampled.size())
        #print("Rank of sampled y is : ", torch.matrix_rank(y_sampled))
        #print(np.sum(np.asmatrix(y_sampled),axis=0))
        #print("Size of output decoder  matrix is : ", V.size())

        V_sampled = torch.index_select(V, 1, sampled_labels)  #indexes the input tensor along column using the entries in indices
        VtV  = torch.mm(V_sampled.t(), V_sampled)               # Label Embedding Matrix
        A  = torch.mm(y_sampled.t(), y_sampled)  			  # models co-occurence of labels
        #print(torch.nonzero(A[2,:]))
        #print("Size of yty is : ", A.size())
        #print(np.sum(np.asmatrix(A),axis=1))
        #print(torch.matrix_rank(A))
        #inp_ae_fp, out_ae_fp, reg_fp = Glas_XC.forward(x, y)
      #  reg_fip = torch.index_select(reg_fp, 1, sampled_labels)
        # Build GLAS Regularizer

        # Sampling the Label matrix per batch done!

        #v  = Glas_XC.encode_output(y)    # Label Embedding Matrix for mini-batch
        #V  = torch.mm(v.t(), v)               # co-occurence in the latent/embedded space
        #A  = torch.mm(y.t(), y)               # models co-occurence of labels

        Z  = torch.diag(A)  #+ epsilon        # returns the diagoan in vector form
        Z  = torch.diag(Z)                    # creates the diagonal from the vector
        #AZ = torch.add(torch.mm(A, torch.pinverse(Z)), torch.mm(torch.pinverse(Z), A)) # to be used for Eurlex4k
        AZ = torch.add(torch.mm(A, torch.inverse(Z)), torch.mm(torch.inverse(Z), A))
        M  = mean*AZ                          # Mean of conditional frequencies of label
        g  = torch.sub(VtV, M)
        gl = torch.norm(g, p='fro')
        loss_glas = div * gl*gl                  # final loss of glas regularizer

        #pos_ind, neg_ind = stocastic_negative_sampling(csr_matrix(y.cpu().numpy()), y.size(1))
        #pos_ind, neg_ind = torch.from_numpy(np.array(pos_ind)).to(cur_device), torch.from_numpy(np.array(neg_ind)).to(cur_device)
        #print (reg_fp.take(neg_ind))
        #reg_fp = reg_fp.cpu()

        # stochastic negative sampling
        mask = torch.ones_like(y)
        mask[y == 1] = 0
        num_neg_sample = 100
        neg_ind = torch.multinomial(mask, num_neg_sample)
        #neg_ind = torch.multinomial(mask * label_count, num_neg_sample)
        neg_ind = neg_ind + torch.arange(0, neg_ind.size(0)).view(-1, 1).to(cur_device) * y.size(1)
        neg_ind = neg_ind.view(-1)

        mask = torch.ones_like(y) 
        mask[y == 0] = 0
        num_pos_sample = 1
        pos_ind = torch.multinomial(mask, num_pos_sample)
        #pos_ind = torch.multinomial(mask / (label_count + 1), num_pos_sample)
        #print (pos_ind)
        pos_ind = pos_ind + torch.arange(0, pos_ind.size(0)).view(-1, 1).to(cur_device) * y.size(1)
        #print (pos_ind.repeat(1, num_neg_sample))
        pos_ind = pos_ind.repeat(1, num_neg_sample).view(-1)
        #print (pos_ind.size())
        #print (pos_ind)
        ###

        margin = 0.01
        class_loss = torch.mean(F.relu(reg_fp.take(neg_ind) - reg_fp.take(pos_ind) + margin))
        loss = class_loss + LAMBDA * loss_glas
        #loss = F.binary_cross_entropy(reg_fp, y) + LAMBDA * loss_glas
        #criterion = torch.nn.BCEWithLogitsLoss()
        #loss = criterion(reg_fp, y) + LAMBDA * loss_glas
        loss.backward()
        #print("Backprop for epoch ", epoch, " done")
        optimizer.step()
        all_iters += 1
        if all_iters % args.interval == 0:
            print("{} / {} :: {} / {} - CLASS_LOSS : {}"
                  .format(epoch, args.epochs, cur_no, len_loader,round(loss.item(), 5)))

        CLASS_LOSS.append(loss.item())

    #scheduler.step()

    if epoch % 5 != 0:
        continue
    with torch.no_grad():
        # build MIPS starts--------------
        #V = Glas_XC.decode_output_weight().t()
        #V = F.normalize(V, p=2, dim=1)
        #V = V.cpu().numpy()

        #index = hnswlib.Index(space = 'cosine', dim = V.shape[1])
        #index.init_index(max_elements = V.shape[0], ef_construction = 200, M = 50)
        #index.add_items(V, np.arange(V.shape[0]))
        #index.set_ef(500)
        #index = buildIndex(V.cpu().numpy())

        pred_y = []
        actual_y = []
        for x, y in iter(train_data_loader):
            x = x.to(device=cur_device, dtype=torch.float)
            pred = Glas_XC.predict(x).argsort(dim=1, descending=True)[:, :5].detach().cpu().numpy()
            #emb = Glas_XC.get_embedding(x)
            #emb = F.normalize(emb, p=2, dim=1)
            #emb = emb.cpu().numpy()
            #pred, _ = index.knn_query(emb, k=5)
            pred_y.append(pred)
            actual_y.append(y)
        res = np.vstack(pred_y)
        actual_y = np.vstack(actual_y)
        actual_y = csr_matrix(actual_y)


        targets = csr2list(actual_y)
        #targets = csr2list(train_loader.classes)
        mlb = MultiLabelBinarizer(range(train_loader.classes.shape[1]), sparse_output=True)
        targets = mlb.fit_transform(targets)
        print('Precision@1,3,5:', get_p_1(res, targets, mlb), get_p_3(res, targets, mlb), get_p_5(res, targets, mlb))
        #print('nDCG@1,3,5:', get_n_1(res, targets, mlb), get_n_3(res, targets, mlb), get_n_5(res, targets, mlb))
        print('PSPrecision@1,3,5:', get_psp_1(res, targets, inv_propen, mlb), get_psp_3(res, targets, inv_propen, mlb),
              get_psp_5(res, targets, inv_propen, mlb))
        #print('PSnDCG@1,3,5:', get_psndcg_1(res, targets, inv_propen, mlb), get_psndcg_3(res, targets, inv_propen, mlb),
        #      get_psndcg_5(res, targets, inv_propen, mlb))

        print ('testing....')
        pred_y = []
        actual_y = []
        for x, y in iter(test_data_loader):
            x = x.to(device=cur_device, dtype=torch.float)
            pred = Glas_XC.predict(x).argsort(dim=1, descending=True)[:, :5]
            pred_y.append(pred.detach().cpu().numpy())
            #emb = Glas_XC.get_embedding(x)
            #emb = F.normalize(emb, p=2, dim=1)
            #emb = emb.cpu().numpy()
            #pred, _ = index.knn_query(emb, k=5)
            #pred_y.append(pred)
            actual_y.append(y)
        res = np.vstack(pred_y)
        actual_y = np.vstack(actual_y)
        actual_y = csr_matrix(actual_y)

        targets = csr2list(actual_y)
        targets = mlb.transform(targets)
        print('Precision@1,3,5:', get_p_1(res, targets, mlb), get_p_3(res, targets, mlb), get_p_5(res, targets, mlb))
        #print('nDCG@1,3,5:', get_n_1(res, targets, mlb), get_n_3(res, targets, mlb), get_n_5(res, targets, mlb))
        print('PSPrecision@1,3,5:', get_psp_1(res, targets, inv_propen, mlb), get_psp_3(res, targets, inv_propen, mlb),
              get_psp_5(res, targets, inv_propen, mlb))
        #print('PSnDCG@1,3,5:', get_psndcg_1(res, targets, inv_propen, mlb), get_psndcg_3(res, targets, inv_propen, mlb),
        #      get_psndcg_5(res, targets, inv_propen, mlb))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Plot graphs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
if args.plot:
    fig = plt.figure(figsize=(9, 18))
    gridspec = gs.GridSpec(4, 6, figure=fig)
    gridspec.tight_layout(fig)
    ax1 = plt.subplot(gridspec[0, :2])
    ax2 = plt.subplot(gridspec[0, 2:4])
    ax3 = plt.subplot(gridspec[0, 4:])
    #ax4 = plt.subplot(gridspec[1:3, 1:5])
    #ax5 = plt.subplot(gridspec[3, :3])
    #ax6 = plt.subplot(gridspec[3, 3:])


    ax1.plot(list(range(1, all_iters + 1)), CLASS_LOSS, 'b', linewidth=2.0)
    ax1.set_title('Classification loss')
    ax2.plot(list(range(1, args.epochs + 1)), AVG_P_AT_K, 'g', linewidth=2.0)
    ax2.set_title('Average Precision at {} (over all datapoints) with epochs'.format(K))
    ax3.plot(list(range(1, args.epochs + 1)), AVG_NDCG_AT_K, 'b', linewidth=2.0)
    ax3.set_title('Average NDCG at {} (over all datapoints) with epochs'.format(K))
    #plt.show()
    plt.savefig('prec_plots.png')
'''

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Save your model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if args.save_model is not None:
    if 'inputAE' in args.save_model or 'all' in args.save_model:
        torch.save(Glas_XC.input_encoder.to('cpu'),
                   'trained_input_encoder_{}.pt'.format(TIME_STAMP))
        torch.save(Glas_XC.input_decoder.to('cpu'),
                   'trained_input_decoder_{}.pt'.format(TIME_STAMP))

    if 'outputAE' in args.save_model or 'all' in args.save_model:
        torch.save(Glas_XC.output_encoder.to('cpu'),
                   'trained_output_encoder_{}.pt'.format(TIME_STAMP))
        torch.save(Glas_XC.output_decoder.to('cpu'),
                   'trained_output_decoder_{}.pt'.format(TIME_STAMP))

    if 'regressor' in args.save_model or 'all' in args.save_model:
        torch.save(Glas_XC.regressor.to('cpu'),
                   'trained_regressor_{}.pt'.format(TIME_STAMP))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Prediction on test ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
