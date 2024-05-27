import os
import time
import argparse
import numpy as np
import random

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
# from tensorboardX import SummaryWriter

import model
import wandb
import lightgcn
import evaluate
import data_utils
from loss import loss_function, PLC_uncertain_discard
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', 
	type = str,
	help = 'dataset used for training, options: amazon_book, yelp, adressa',
	default = 'adressa')
parser.add_argument('--model', 
	type = str,
	help = 'model used for training. options: GMF, NeuMF-end',
	default = 'NeuMF-end')
parser.add_argument('--drop_rate', 
	type = float,
	help = 'drop rate',
	default = 0.2)
parser.add_argument('--num_gradual', 
	type = int, 
	default = 30000,
	help='how many epochs to linearly increase drop_rate')
parser.add_argument('--exponent', 
	type = float, 
	default = 1, 
	help='exponent of the drop rate {0.5, 1, 2}')
parser.add_argument("--lr", 
	type=float, 
	default=0.001, 
	help="learning rate")
parser.add_argument("--dropout", 
	type=float,
	default=0.0,  
	help="dropout rate")
parser.add_argument("--batch_size", 
	type=int, 
	default=1024, 
	help="batch size for training")
parser.add_argument("--epochs", 
	type=int,
	default=10,
	help="training epoches")
parser.add_argument("--eval_freq", 
	type=int,
	default=2000,
	help="the freq of eval")
parser.add_argument("--top_k", 
	type=list, 
	default=[5, 10, 20, 50],
	help="compute metrics@top_k")
parser.add_argument("--factor_num", 
	type=int,
	default=32, 
	help="predictive factors numbers in the model")
parser.add_argument("--num_layers", 
	type=int,
	default=3, 
	help="number of layers in MLP model")
parser.add_argument("--num_ng", 
	type=int,
	default=1, 
	help="sample negative items for training")
parser.add_argument("--out", 
	default=True,
	help="save model or not")
parser.add_argument("--gpu", 
	type=str,
	default="1",
	help="gpu card ID")
parser.add_argument('--time_step', 
	type=int, 
	default=3, 	
	help='time_step')
parser.add_argument('--co_lambda', 
	type=float, 
	help='sigma^2', 
	default=1e-4)
parser.add_argument('--epoch_decay_start', 
	type=int, 
	default=10)
parser.add_argument("--relabel_ratio", 
	type=float,
	default=0.05,
	help="relabel ratio")
args = parser.parse_args()



wandb.login
wandb.init(project="DCF", config=args, notes='DCF')

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True

torch.manual_seed(2019) # cpu
torch.cuda.manual_seed(2019) #gpu
np.random.seed(2019) #numpy
random.seed(2019) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn

def worker_init_fn(worker_id):
    np.random.seed(2019 + worker_id)

sys.stdout = open('../loggg.log', 'a')
data_path = '../data/{}/'.format(args.dataset)
model_path = '../models/{}/'.format(args.dataset)
print("arguments: %s " %(args))
print("config model", args.model)
print("config data path", data_path)
print("config model path", model_path)

############################## PREPARE DATASET ##########################

train_data, valid_data, test_data_pos, user_pos, user_num ,item_num, train_mat, train_data_noisy = data_utils.load_all(args.dataset, data_path)

train_adj = data_utils.create_adj_mat(train_mat, user_num, item_num)

# construct the train and test datasets
train_dataset = data_utils.NCFData(
		train_data, item_num, train_mat, args.num_ng, 0, train_data_noisy)
valid_dataset = data_utils.NCFData(
		valid_data, item_num, train_mat, args.num_ng, 1)

train_loader = data.DataLoader(train_dataset,
		batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
valid_loader = data.DataLoader(valid_dataset,
		batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

print("data loaded! user_num:{}, item_num:{} train_data_len:{} test_user_num:{}".format(user_num, item_num, len(train_data), len(test_data_pos)))

########################### CREATE MODEL #################################
GMF_model = None
MLP_model = None

if args.model == 'GMF':
    model = model.NCF(user_num, item_num, args.factor_num, args.num_layers, 
                    args.dropout, args.model, GMF_model, MLP_model)
elif args.model == 'NeuMF-end':
    model = model.NCF(user_num, item_num, args.factor_num, args.num_layers, 
                    args.dropout, args.model, GMF_model, MLP_model)
elif args.model == 'CDAE':
    train_matrix = torch.zeros([user_num, item_num])
    for (u, i) in train_mat.keys():
        train_matrix[u, i] = 1.0
    model = model.CDAE(user_num, item_num, hidden_dim=32, device="cuda",
                 corruption_ratio=0.5, act='tanh')
    train_matrix=train_matrix.cuda()
else:
    model = model.LightGCN(user_num, item_num, train_adj, args.factor_num, args.num_layers)


model.cuda()
BCE_loss = nn.BCEWithLogitsLoss()
co_lambda_plan = args.co_lambda * np.linspace(1, 0, args.epochs) 

if args.model == 'NeuMF-pre':
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
else:
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


def drop_rate_schedule(iteration):
	drop_rate = np.linspace(0, args.drop_rate**args.exponent, args.num_gradual)
	if iteration < args.num_gradual:
		return drop_rate[iteration]
	else:
		return args.drop_rate


########################### Eval #####################################
def eval(model, valid_loader, best_loss, count):
    	
    model.eval()
    epoch_loss = 0
    valid_loader.dataset.ng_sample() # negative sampling
    for user, item, label, noisy_or_not in valid_loader:
        user = user.cuda()
        item = item.cuda()
        label = label.float().cuda()

        prediction = model(user, item)
        loss = BCE_loss(prediction, label)
        # loss = loss_function(prediction, label, drop_rate_schedule(count))
        epoch_loss += loss.detach()
    print("################### EVAL ######################")
    print("Eval loss:{}".format(epoch_loss))
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        if args.out:
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            torch.save(model, '{}{}_{}-{}.pth'.format(model_path, args.model, args.drop_rate, args.num_gradual))
    return best_loss

########################### Test #####################################
def test(model, test_data_pos, user_pos):
	top_k = args.top_k
	model.eval()
	# The second item is too large, and there may be a problem with vacant values in predictions.
	_, recall, NDCG, _ = evaluate.test_all_users(model, 256, item_num, test_data_pos, user_pos, top_k)

	print("################### TEST ######################")
	print("Recall {:.4f}-{:.4f}-{:.4f}-{:.4f}".format(recall[0], recall[1], recall[2], recall[3]))
	print("NDCG {:.4f}-{:.4f}-{:.4f}-{:.4f}".format(NDCG[0], NDCG[1], NDCG[2], NDCG[3]))
	wandb.log({
	"Recall@5": recall[0],
	"Recall@10": recall[1],
	"Recall@20": recall[2],
	"Recall@50": recall[3],
	"NDCG@5": NDCG[0],
	"NDCG@10": NDCG[1],
	"NDCG@20": NDCG[2],
	"NDCG@50": NDCG[3]})

########################### TRAINING #####################################
count, best_hr = 0, 0
best_loss = 1e9
best_loss_2 = 1e9


for epoch in range(args.epochs):
	if epoch % args.time_step == 0:
		print('Time step initializing...')
		before_loss = 0.0 * np.ones((len(train_dataset), 1))
		sn = torch.from_numpy(np.ones((len(train_dataset), 1)))
	model.train()

	start_time = time.time()
	train_loader.dataset.ng_sample()
	before_loss_list=[]	
	ind_update_list = []

	for i, (user, item, label, noisy_or_not) in enumerate(train_loader):
		user = user.cuda()
		item = item.cuda()
		start_point = int(i * args.batch_size)
		stop_point = int((i + 1) * args.batch_size)

		label = torch.tensor(train_mat[user.cpu().numpy().tolist(), item.cpu().numpy().tolist()].todense()).squeeze().cuda()

		model.zero_grad()
		if args.model == 'CDAE':
			prediction = model(user, train_mat) 
		else:
			prediction = model(user, item)

		loss, train_adj, loss_mean, ind_update = PLC_uncertain_discard(user, item, train_mat, prediction, label, drop_rate_schedule(count), epoch, sn[start_point:stop_point], before_loss[start_point:stop_point], co_lambda_plan[epoch], args.relabel_ratio)
		before_loss_list += list(np.array(loss_mean.detach().cpu()))
		ind_update_list += list(np.array(ind_update.cpu() + i * args.batch_size))
  
		train_mat = train_adj
		loss.backward()
		optimizer.step()
		wandb.log({'drop_rate' : drop_rate_schedule(count)})
		if count % args.eval_freq == 0 and count != 0:
			print("epoch: {}, iter: {}, loss:{}".format(epoch, count, loss))
			best_loss = eval(model, valid_loader, best_loss, count)
			model.train()			
			wandb.log({
			"epoch": epoch,
			"iter": count,
			"loss": loss})
		count += 1
	before_loss = np.array(before_loss_list).astype(float)
	all_zero_array = np.zeros((len(train_dataset), 1))
	all_zero_array[np.array(ind_update_list)] = 1
	sn += torch.from_numpy(all_zero_array)

print("############################## Training End. ##############################")
test_model = torch.load('{}{}_{}-{}.pth'.format(model_path, args.model, args.drop_rate, args.num_gradual))
test_model.cuda()
test(test_model, test_data_pos, user_pos)
