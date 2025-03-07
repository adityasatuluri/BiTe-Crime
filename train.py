import numpy as np
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import glob
from utils import *
from model import *
from layers import *
from sklearn.preprocessing import MinMaxScaler
import shutil
import sys
import argparse as Ap
from datetime import date, datetime
import os

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d--%H-%M-%S")

seed = 0x6a09e667f3bcc908
np.random.normal(seed & 0xFFFFFFFF)
torch.manual_seed(seed & 0xFFFFFFFF)

directory = "result/training_logs"
file_date = directory + "/exec" + str(formatted_datetime) + ".txt"
os.makedirs(directory, exist_ok=True)
# execlog = open(file_date, 'a')

argp = Ap.ArgumentParser()
argp.add_argument("--tct", default='chicago', type=str, help="Target city")
argp.add_argument("--tr", default=1, type=int, help="Target region")
argp.add_argument("--tc", default=1, type=int, help="Target category")
argp.add_argument("--bs", default=42, type=int, help="Batch size")
argp.add_argument("--ts", default=120, type=int, help="Number of time steps")
argp.add_argument("--rts", default=24, type=int, help="Number of recent time steps")
argp.add_argument("--ncf", default=1, type=int, help="Number of crime features per time step")
argp.add_argument("--nxf", default=12, type=int, help="Number of external features per time step")
argp.add_argument("--gout", default=16, type=int, help="Dimension of output features of GAT")
argp.add_argument("--gatt", default=64, type=int, help="Dimension of attention module of GAT")
argp.add_argument("--rhid", default=64, type=int, help="Dimension of hidden state of SAB-LSTMs")
argp.add_argument("--ratt", default=50, type=int, help="Dimension of attention module of SAB-LSTMs")
argp.add_argument("--rl", default=1, type=int, help="Number of layers of SAB-LSTMs")
d = argp.parse_args(sys.argv[1:])

target_city = d.tct
target_region = d.tr
target_cat = d.tc
time_step = d.ts
recent_time_step = d.rts
batch_size = d.bs
gat_out = d.gout
gat_att = d.gatt
ncfeature = d.ncf
nxfeature = d.nxf
slstm_nhid = d.rhid
slstm_nlayer = d.rl
slstm_att = d.ratt

# execlog.write(f"Target City: {d.tct}\n"
#               f"Target Region: {d.tr}\n"
#               f"Target Category: {d.tc}\n"
#               f"Time Step: {d.ts}\n"
#               f"Recent Time Step: {d.rts}\n"
#               f"Batch Size: {d.bs}\n"
#               f"GAT Output: {d.gout}\n"
#               f"GAT Attention: {d.gatt}\n"
#               f"Number of Categorical Features: {d.ncf}\n"
#               f"Number of Numerical Features: {d.nxf}\n"
#               f"Spatial LSTM Hidden Units: {d.rhid}\n"
#               f"Spatial LSTM Layers: {d.rl}\n"
#               f"Spatial LSTM Attention: {d.ratt}\n\n\n")

gen_gat_adj_file(target_city, target_region)
loaded_data = torch.from_numpy(np.loadtxt("data/" + target_city + "/com_crime/r_" + str(target_region) + ".txt", dtype=int)).T
loaded_data = loaded_data[:, target_cat:target_cat+1]

# execlog.write(f"loaded_data: {loaded_data}\n\n\n")

x, y, x_daily, x_weekly = create_inout_sequences(loaded_data)
# execlog.write(f"x: {x}, y: {y}, x_daily = {x_daily}, x_weekly = {x_weekly}")

scale = MinMaxScaler(feature_range=(-1, 1))
x = torch.from_numpy(scale.fit_transform(x))
x_daily = torch.from_numpy(scale.fit_transform(x_daily))
x_weekly = torch.from_numpy(scale.fit_transform(x_weekly))
y = torch.from_numpy(scale.fit_transform(y))

# execlog.write(f"After Transformation:\n"
#               f"x: {x}, y: {y}, x_daily = {x_daily}, x_weekly = {x_weekly}")

train_x_size = int(x.shape[0] * .67)
train_x = x[: train_x_size, :]
train_x_daily = x_daily[: train_x_size, :]
train_x_weekly = x_weekly[: train_x_size, :]
train_y = y[: train_x_size, :]
test_x = x[train_x_size:, :]
test_x_daily = x_daily[train_x_size:, :]
test_x_weekly = x_weekly[train_x_size:, :]
test_x = test_x[:test_x.shape[0] - 11, :]
test_x_daily = test_x_daily[:test_x_daily.shape[0] - 11, :]
test_x_weekly = test_x_weekly[:test_x_weekly.shape[0] - 11, :]
test_y = y[train_x_size:, :]
test_y = test_y[:test_y.shape[0] - 11, :]

train_x = train_x.view(int(train_x.shape[0] / batch_size), batch_size, time_step)
train_x_daily = train_x_daily.view(int(train_x_daily.shape[0] / batch_size), batch_size, train_x_daily.shape[1])
train_x_weekly = train_x_weekly.view(int(train_x_weekly.shape[0] / batch_size), batch_size, train_x_weekly.shape[1])
train_y = train_y.view(int(train_y.shape[0] / batch_size), batch_size, 1)
test_x = test_x.view(int(test_x.shape[0] / batch_size), batch_size, time_step)
test_x_daily = test_x_daily.view(int(test_x_daily.shape[0] / batch_size), batch_size, test_x_daily.shape[1])
test_x_weekly = test_x_weekly.view(int(test_x_weekly.shape[0] / batch_size), batch_size, test_x_weekly.shape[1])
test_y = test_y.view(int(test_y.shape[0] / batch_size), batch_size, 1)

train_feat, test_feat = load_data_regions(batch_size, target_cat, target_region, target_city)
train_feat_ext, test_feat_ext = load_data_regions_external(batch_size, nxfeature, target_region, target_city)
train_crime_side, test_crime_side = load_data_sides_crime(batch_size, target_cat, target_region, target_city)

model = AIST(ncfeature, nxfeature, gat_out, gat_att, slstm_nhid, slstm_att, slstm_nlayer, batch_size,
             recent_time_step, target_city, target_region, target_cat)
n = sum(p.numel() for p in model.parameters() if p.requires_grad)
# execlog.write(f"n - model parameters:\n{n}")

lr = 0.001
weight_decay = 5e-4
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
criterion = nn.L1Loss()

epochs = 300
best = float('inf')
best_epoch = 0
t_total = time.time()
train_batch = train_x.shape[0]
test_batch = test_x.shape[0]

class EarlyStopping:
    def __init__(self, patience=20, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.counter = 0

    def save_checkpoint(self, val_loss):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        self.val_loss_min = val_loss

early_stopping = EarlyStopping(patience=20, verbose=True)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for i in range(train_batch):
        t = time.time()
        x_crime = Variable(train_x[i]).float()
        x_crime_daily = Variable(train_x_daily[i]).float()
        x_crime_weekly = Variable(train_x_weekly[i]).float()
        y = Variable(train_y[i]).float()
        
        optimizer.zero_grad()
        output, attn = model(x_crime, x_crime_daily, x_crime_weekly, train_feat[i], train_feat_ext[i], train_crime_side[i])
        y = y.view(-1, 1)
        loss_train = criterion(output, y)
        loss_train.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss_train.item()
        print('Epoch: {:04d}'.format(epoch*train_batch + i + 1),
              'loss_train: {:.4f}'.format(loss_train.data.item()),
              'time: {:.4f}s'.format(time.time() - t))

    avg_loss = total_loss / train_batch
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i in range(test_batch):
            x_crime_test = Variable(test_x[i]).float()
            x_crime_daily_test = Variable(test_x_daily[i]).float()
            x_crime_weekly_test = Variable(test_x_weekly[i]).float()
            y_test = Variable(test_y[i]).float()
            output_test, _ = model(x_crime_test, x_crime_daily_test, x_crime_weekly_test, test_feat[i], test_feat_ext[i], test_crime_side[i])
            y_test = y_test.view(-1, 1)
            loss_test = criterion(output_test, y_test)
            val_loss += loss_test.item()
    
    avg_val_loss = val_loss / test_batch
    
    # Early stopping
    early_stopping(avg_val_loss)
    if early_stopping.early_stop:
        print("Early stopping")
        break
    
    # Learning rate scheduler step
    scheduler.step(avg_val_loss)
    
    # Save model if it's the best so far
    if avg_val_loss < best:
        best = avg_val_loss
        best_epoch = epoch
        torch.save(model.state_dict(), f'best_model_epoch_{epoch}.pkl')

    # Remove older model files
    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('_')[-1].split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

    if epoch*train_batch + i + 1 >= 300:
        break

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

print(f'Loading best model from epoch {best_epoch}')
model.load_state_dict(torch.load(f'best_model_epoch_{best_epoch}.pkl'))

f = open('result/aist_new.txt','a')
stat_y = []
stat_y_prime = []

def compute_test():
    loss = 0
    model.eval()
    with torch.no_grad():
        for i in range(test_batch):
            x_crime_test = Variable(test_x[i]).float()
            x_crime_daily_test = Variable(test_x_daily[i]).float()
            x_crime_weekly_test = Variable(test_x_weekly[i]).float()
            y_test = Variable(test_y[i]).float()
            output_test, list_att = model(x_crime_test, x_crime_daily_test, x_crime_weekly_test, test_feat[i], test_feat_ext[i], test_crime_side[i])
            y_test = y_test.view(-1, 1)
            y_test = torch.from_numpy(scale.inverse_transform(y_test))
            output_test = torch.from_numpy(scale.inverse_transform(output_test.detach()))
            stat_y.append(y_test.detach().numpy())
            stat_y_prime.append(output_test.numpy())
            loss_test = criterion(output_test, y_test)
            loss += loss_test.data.item()
    
    print("Test set results:", "loss= {:.4f}".format(loss / test_batch))
    print("Target Region: ", target_region, " Target Category: ", target_cat, " Loss/i: ", loss / test_batch)
    print(target_region, " ", target_cat, " ", loss/test_batch, file=f)
    #print("Date: ", date.today(), " Target Region: ", target_region, " Target Category: ", target_cat, " Loss/i: ", loss / test_batch, file=f)
    #print("Date: ", date.today(), " Target Region: ", target_region, " Target Category: ", target_cat, " Loss/i: ", loss / test_batch, file=execlog)

compute_test()
