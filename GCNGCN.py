# GCN for Thermal analysis

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, time, os
import numpy as np
from numpy import genfromtxt

import csv
from shutil import copyfile

dataset_dir = './newdata/'




MaxMinValues = genfromtxt(dataset_dir + 'MaxMinValues.csv', delimiter=',')

tPower_max = MaxMinValues[1, 0]
tPower_min = MaxMinValues[1, 1]
Power_max = MaxMinValues[1, 2]
Power_min = MaxMinValues[1, 3]
Temperature_max = MaxMinValues[1, 4]
Temperature_min = MaxMinValues[1, 5]
Conductance_max = MaxMinValues[1, 6]
Conductance_min = MaxMinValues[1, 7]

print(tPower_max, tPower_min, Power_max,Power_min,Temperature_max,Temperature_min,Conductance_max,Conductance_min)

last_saved_epoch = 0
date = '20210225'
dir_name = 'GCN'
ckpt_dir = 'ckpt/{}_{}'.format(dir_name, date)
is_inference = True
ckpt_file = ckpt_dir + '/HSgcn_92.pkl'

n_hidden_n = [1, 16, 32, 64, 128, 256, 512, 512, 512, 256, 128, 64, 32, 16, 1]#[1, 16, 32,32,  64,64, 128,128, 256,256, 512,512,  1024,  512,512, 256,256,128, 128, 64, 64]
e_hidden_e = [1, 16, 32, 64, 128, 256, 512, 512, 512, 256, 128, 64, 32, 16, 0]#[1, 16, 32,32,  64,64, 128,128, 256,256, 512,512,  1024,  512,512, 256,256,128, 128, 64, 0]

#MLP_hidden_n = [64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024, 2048,2048, 4096, 4096, 2048, 2048, 1024,1024, 512,512, 256,256, 128,128, 64,64,  1]


batch_size = 2

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Running on GPU!')
else:
    device = torch.device('cpu')
    print('Runing on CPU!')

#device = torch.device('cpu')

class HSConv(nn.Module):
    def __init__(self,
                 Skipnode_in_feats_size,
                 node_in_feats_size,
                 edge_in_feats_size,
                 node_out_feats_size,
                 edge_out_feats_size):
        super(HSConv, self).__init__()

        self.Skipnode_in_feats_size = Skipnode_in_feats_size
        self.node_in_feats_size = node_in_feats_size
        self.edge_in_feats_size = edge_in_feats_size
        self.node_out_feats_size = node_out_feats_size
        self.edge_out_feats_size = edge_out_feats_size

        self.weight_n2n_u = nn.Parameter(torch.Tensor(self.node_in_feats_size,self.node_out_feats_size))
        self.weight_n2n_v = nn.Parameter(torch.Tensor(self.Skipnode_in_feats_size+self.node_in_feats_size + self.node_out_feats_size,self.node_out_feats_size))
        self.weight_e2n = nn.Parameter(torch.Tensor(self.edge_in_feats_size,self.node_out_feats_size))
        self.bias_n = nn.Parameter(torch.Tensor(self.node_out_feats_size))

        if self.edge_out_feats_size != 0:
            self.weight_n2e_u = nn.Parameter(torch.Tensor(self.node_in_feats_size,self.edge_out_feats_size))
            self.weight_n2e_v = nn.Parameter(torch.Tensor(self.Skipnode_in_feats_size + self.node_in_feats_size,self.edge_out_feats_size))
            self.weight_e2e = nn.Parameter(torch.Tensor(self.edge_in_feats_size + self.edge_out_feats_size,self.edge_out_feats_size))
            self.bias_e = nn.Parameter(torch.Tensor(self.edge_out_feats_size))
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv1 = 0.2 / math.sqrt(self.weight_n2n_u.size(1))
        self.weight_n2n_u.data.uniform_(-stdv1,stdv1)
        stdv2 = 0.2 / math.sqrt(self.weight_n2n_v.size(1))
        self.weight_n2n_v.data.uniform_(-stdv2,stdv2)
        stdv3 = 0.2 / math.sqrt(self.weight_e2n.size(1))
        self.weight_e2n.data.uniform_(-stdv3,stdv3)
        stdv4 = 0.2 / math.sqrt(self.bias_n.size(0))
        self.bias_n.data.uniform_(-stdv4, stdv4)
        
        if self.edge_out_feats_size != 0:
            stdv5 = 0.2 / math.sqrt(self.weight_n2e_u.size(1))
            self.weight_n2e_u.data.uniform_(-stdv5,stdv5)
            stdv6 = 0.2 / math.sqrt(self.weight_n2e_v.size(1))
            self.weight_n2e_v.data.uniform_(-stdv6,stdv6)
            stdv7 = 0.2 / math.sqrt(self.weight_e2e.size(1))
            self.weight_e2e.data.uniform_(-stdv7,stdv7)
            stdv8 = 0.2 / math.sqrt(self.bias_e.size(0))
            self.bias_e.data.uniform_(-stdv8, stdv8)
            
    def forward(self, g, Skipnode_in_feats, node_in_feats, edge_in_feats):
        with g.local_scope():
            g.ndata['h1'] = torch.mm(node_in_feats,self.weight_n2n_u)
            g.edata['e1'] = torch.mm(edge_in_feats,self.weight_e2n)
            g.update_all(fn.u_add_e('h1','e1','m'),fn.mean('m','h'))
            h_neigh  = g.ndata['h']
            h_total = torch.cat([Skipnode_in_feats, node_in_feats, h_neigh], dim=1)
            g.ndata['h'] = torch.mm(h_total, self.weight_n2n_v) + self.bias_n
            
            if self.edge_out_feats_size != 0:
                g.srcdata['hu'] = torch.mm(node_in_feats,self.weight_n2e_u)
                hc = torch.cat([Skipnode_in_feats, node_in_feats], dim=1)
                g.dstdata['hv'] = torch.mm(hc,self.weight_n2e_v)
                g.apply_edges(fn.u_add_v('hu', 'hv', 'e2'))
                e_neigh  = g.edata['e2']
                e_total = torch.cat([edge_in_feats, e_neigh], dim=1)
                g.edata['e'] = torch.mm(e_total, self.weight_e2e) + self.bias_e
                
            if self.edge_out_feats_size != 0:
                return g.ndata['h'], g.edata['e']
            else:
                return g.ndata['h']

class HSModel(nn.Module):
    def __init__(self,
                 Skipnode_in_feats_size,
                 n_hidden_n,
                 e_hidden_e,
                 activation):
        super(HSModel, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.n_layers = len(n_hidden_n)
        self.edge_out_feats_size = e_hidden_e[-1]

        for i in range(self.n_layers - 1):
            self.layers.append(HSConv( Skipnode_in_feats_size, n_hidden_n[i], e_hidden_e[i], n_hidden_n[i + 1], e_hidden_e[i + 1]))
            
    def forward(self, g, Skipnode_in_feats, node_in_feats, edge_in_feats):
        hv = node_in_feats
        he = edge_in_feats
        for l, layer in enumerate(self.layers):
            if l == len(self.layers) - 1 and self.edge_out_feats_size == 0:
                hv = layer(g, Skipnode_in_feats, hv, he)
                return hv

            hv, he = layer(g, Skipnode_in_feats, hv, he)
            if l != len(self.layers) - 1:
                hv = self.activation(hv)
                he = self.activation(he)

        return hv, he

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class MLP(nn.Module):
    def __init__(self,
                 skipnode_in_feats_size,
                 MLP_hidden_n):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(MLP_hidden_n) - 1):
            self.layers.append(nn.Linear(skipnode_in_feats_size+MLP_hidden_n[i], MLP_hidden_n[i + 1]))
        self.weight_init(mean=0.0, std=0.02)

    def weight_init(self, mean, std):
        for m in self.layers:
            normal_init(m, mean, std)

    def forward(self, skipnode_in_feats, hv):
        for l, m in enumerate(self.layers):
            hc = torch.cat([skipnode_in_feats,hv], dim=1)
            hv = m(hc)
            if l != len(self.layers) - 1:
                hv = F.relu(hv)

        return hv 
        
def evaluate(model, g, node_feats1, node_feats2, node_labels, edge_features):
    model.eval()
   # node_MLP.eval()
    with torch.no_grad():
        #hc = torch.cat([node_feats1, node_feats2], dim=1)
        nt = model(g, node_feats2, node_feats1, edge_features)
        #nt = node_MLP(node_feats2,nt)
    
        
        err = torch.sum((node_labels - nt)** 2)
        
        length = nt.shape[0] * nt.shape[1]
        
        rmse = torch.sqrt(err / length)

        MAE = torch.mean(torch.abs(node_labels-nt))

        AEmax = torch.max(torch.abs(node_labels-nt))

        AEmin = torch.min(torch.abs(node_labels-nt))

        return rmse, nt, MAE, AEmax, AEmin

def read_node(train_batch):
    node_feats1_list = []
    node_feats2_list = []
    node_labels_list = []
    for case_num in train_batch:
        node_feats_read1 = genfromtxt(dataset_dir + 'Power_{}.csv'.format(case_num), delimiter=',')
        node_feats_read2 = genfromtxt(dataset_dir + 'totalPower_{}.csv'.format(case_num), delimiter=',')
        node_feats1 = [[0] for _ in range(len(node_feats_read1))]
        for i in range(len(node_feats_read1)):
            node_feats1[i][0] = (node_feats_read1[i,1]-Power_min)/(Power_max - Power_min)*2-1
        node_feats2 = [[0] for _ in range(len(node_feats_read2))]
        for i in range(len(node_feats_read2)):
            node_feats2[i][0] = (node_feats_read2[i]-tPower_min)/(tPower_max - tPower_min)*2-1
        node_feats1_list.append(node_feats1)
        node_feats2_list.append(node_feats2)
        node_labels_read = genfromtxt(dataset_dir + 'Temperature_{}.csv'.format(case_num), delimiter=',')
        node_labels = [[0] for _ in range(len(node_labels_read))]
        for i in range(len(node_labels_read)):
            node_labels[i][0] = node_labels_read[i, 1]
        node_labels = (node_labels - Temperature_min) / (Temperature_max - Temperature_min) * 2 - 1
        node_labels_list.append(node_labels)

    batched_node_feats1 = np.vstack(node_feats1_list)
    batched_node_feats2 = np.vstack(node_feats2_list)
    batched_node_labels = np.vstack(node_labels_list)

    return batched_node_feats1,batched_node_feats2,batched_node_labels

def read_edge(train_batch):
    g_list = []
    edge_feats_list = []
    for case_num in train_batch:
        edge_data = genfromtxt(dataset_dir+'Edge_{}.csv'.format(case_num), delimiter=',')

        g = dgl.graph((edge_data[:, 0].astype(int), edge_data[:, 1].astype(int)))
        edge_feats = [[0] for _ in range(len(edge_data))]
        for i in range(len(edge_data)):
            edge_feats[i][0] = edge_data[i, 1]
        edge_feats = (edge_feats - Conductance_min) / (Conductance_max - Conductance_min) * 2 - 1

        g_list.append(g)
        edge_feats_list.append(edge_feats)
    
    batched_graph = dgl.batch(g_list)
    batched_edge_feats = np.vstack(edge_feats_list)

    return batched_graph, batched_edge_feats


def main():
    global last_saved_epoch
    global Test_Acc_min
    Test_Acc_min = 1

    reader = csv.reader(open(dataset_dir + 'train_data.csv', "r"), delimiter=",")
    train_data = list(reader)
    train_data = np.array(train_data)
    train_data = np.random.permutation(train_data)
    train_data = train_data.reshape(-1).tolist()
    num_train = len(train_data)

    reader = csv.reader(open(dataset_dir + 'test_data.csv', "r"), delimiter=",")
    test_data = list(reader)
    test_data = np.array(test_data)
    test_data = np.random.permutation(test_data)
    test_data = test_data.reshape(-1).tolist()
    num_test = len(test_data)

    model = HSModel(1, n_hidden_n, e_hidden_e, F.relu)
    #node_MLP = MLP(1,MLP_hidden_n)

    if not is_inference:
        model.to(device=device)
        #node_MLP.to(device=device)
        
        
        MSEloss = nn.MSELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        #optimizer_MLP = torch.optim.Adam(node_MLP.parameters(), lr=1e-4)
        
        print('batch size: {}, num_train: {}'.format(batch_size, num_train))

        dur=[0]
        
        for epoch in range(30000):

            train_data = np.array(train_data)
            train_data = np.random.permutation(train_data)
            train_data = train_data.reshape(-1).tolist()

            model.train()
            #node_MLP.train()

            epoch_loss = 0
            epoch_acc = 0

            if epoch >=3:
                t0 = time.time()
            for i in range(0, num_train, batch_size):  
                train_batch = train_data[i : min(i+batch_size, num_train)]
                g, edge_feats = read_edge(train_batch)
                node_feats1, node_feats2, node_labels = read_node(train_batch)

                g = g.to(device=device)
                node_feats1 = torch.Tensor(node_feats1).to(device=device)
                node_feats2 = torch.Tensor(node_feats2).to(device=device)
                node_labels = torch.Tensor(node_labels).to(device=device)
                edge_feats = torch.Tensor(edge_feats).to(device=device)

                #hc = torch.cat([node_feats1,node_feats2], dim=1)
                hv = model(g, node_feats2, node_feats1, edge_feats)

                # loss = MSEloss(hv,node_labels)+MSEloss(he,edge_labels)
                loss = MSEloss(hv,node_labels)
                
                optimizer.zero_grad()
                #optimizer_MLP.zero_grad()

                loss.backward()
                optimizer.step()
                #optimizer_MLP.step()


                acc_current, node_output = evaluate(model, g, node_feats1, node_feats2,node_labels, edge_feats)
                epoch_acc += acc_current
                epoch_loss += loss.item()
                
                #Temperature_output = open('./newdataGCN/Temperature_output_{}.csv'.format('0_0'),'w')
                #for m in range(node_output.shape[0]):
                #    Temperature_output.write(str(m)+","+str((node_output[m][0].tolist()+1)/2.0*(Temperature_max - Temperature_min)+Temperature_min)+"\n")
                
                #Temperature_output.close()

                #Temperature_labels = open('./newdataGCN/Temperature_labels_{}.csv'.format('0_0'),'w')
                #for m in range(node_labels.shape[0]):
                #    Temperature_labels.write(str(m)+","+str((node_labels[m][0].tolist()+1)/2.0*(Temperature_max - Temperature_min)+Temperature_min)+"\n")
                
                #Temperature_labels.close()
                
                #Temperature_base = open('./newdataGCN/Temperature_base_{}.csv'.format('0_0'),'w')
                #for m in range(node_feats2.shape[0]):
                #    Temperature_base.write(str(m)+","+str((node_feats2[m][0].tolist()+1)/2.0*(Temperature_max - Temperature_min)+Temperature_min)+"\n")
                
                #Temperature_base.close()

                if i+2*batch_size >= num_train and i+batch_size<num_train:
                    length_pred = int(math.sqrt((node_output.shape[0]/batch_size - 12)/3))
                    length_stan = int(math.sqrt((node_labels.shape[0]/batch_size - 12)/3))
                    for t in range(1):
                        with open(dataset_dir+'Chiplet.grid.steady','w') as grid:
                            for m in range(length_pred):
                                for n in range(length_pred):
                                    grid.write(str(m*length_pred+n)+" "+ str((node_output[m*length_pred+n][0].tolist()+1)/2.0*(Temperature_max -Temperature_min)+Temperature_min)+"\n")
                                grid.write("\n")
                        cmd = "../grid_thermal_map.pl Chiplet_Core"+ train_batch[t][:train_batch[t].find('_')]+".flp "+dataset_dir+"Chiplet.grid.steady "+str(length_pred)+" "+ str(length_pred)+" > "+dataset_dir+"train_Chiplet_pred"+str(epoch)+"_"+str(0)+".svg"
            
                        os.system(cmd)
    
                        with open(dataset_dir+'Chiplet.grid.steady','w') as grid:
                            for m in range(length_stan):
                                for n in range(length_stan):
                                    grid.write(str(m*length_stan+n)+" "+ str((node_labels[m*length_stan+n][0].tolist()+1)/2.0*(Temperature_max -Temperature_min)+Temperature_min)+"\n")
                                grid.write("\n")
                        cmd = "../grid_thermal_map.pl Chiplet_Core"+ train_batch[t][:train_batch[t].find('_')]+".flp "+dataset_dir+"Chiplet.grid.steady "+str(length_stan)+" "+ str(length_stan)+" > "+dataset_dir+"train_Chiplet_stan"+str(epoch)+"_"+str(0)+".svg"
            
                        os.system(cmd)

                print(epoch,i)
                
            if epoch >=3:
                dur.append(time.time() - t0)

            epoch_acc /= int(num_train/batch_size) + (num_train % batch_size > 0)
            epoch_loss /= int(num_train / batch_size) + (num_train % batch_size > 0)
            
            print("Train Epoch {:05d} |Time(s) {:.4f} | Loss {:04f} | Accuracy {:.4f}".format(epoch, np.mean(dur), epoch_loss, epoch_acc))
            with open(dataset_dir + 'train_acc.txt','a') as Train_Acc_file:
                Train_Acc_file.write("{:05d} {:04f} {:.4f}\n".format(epoch, epoch_loss, epoch_acc))
            

            if epoch % 1 == 0:
                epoch_test_acc = 0
                for i in range(0, num_test, batch_size):  
                    test_batch = test_data[i : min(i+batch_size, num_test)]
                    g, edge_feats = read_edge(test_batch)
                    node_feats1, node_feats2, node_labels = read_node(test_batch)

                    g = g.to(device=device)
                    node_feats1 = torch.Tensor(node_feats1).to(device=device)
                    node_feats2 = torch.Tensor(node_feats2).to(device=device)
                    node_labels = torch.Tensor(node_labels).to(device=device)
                    edge_feats = torch.Tensor(edge_feats).to(device=device)

                    
                        
                        
                    acc, node_output = evaluate(model,  g, node_feats1,node_feats2,node_labels,edge_feats)
                    
                    epoch_test_acc += acc

                    if i+2*batch_size >= num_test and i+batch_size < num_test:
                        with open(dataset_dir+'Chiplet.grid.steady','w') as grid:
                            for m in range(length_pred):
                                for n in range(length_pred):
                                    grid.write(str(m*length_pred+n)+" "+ str((node_output[m*length_pred+n][0].tolist()+1)/2.0*(Temperature_max -Temperature_min)+Temperature_min)+"\n")
                                grid.write("\n")
                        cmd = "../grid_thermal_map.pl Chiplet_Core"+ test_batch[0][:test_batch[0].find('_')]+".flp "+dataset_dir+"Chiplet.grid.steady "+str(length_pred)+" "+ str(length_pred)+" > "+dataset_dir+"test_Chiplet_pred"+str(epoch)+"_"+str(0)+".svg"
                
                        os.system(cmd)
    
                        with open(dataset_dir+'Chiplet.grid.steady','w') as grid:
                            for m in range(length_stan):
                                for n in range(length_stan):
                                    grid.write(str(m*length_stan+n)+" "+ str((node_labels[m*length_stan+n][0].tolist()+1)/2.0*(Temperature_max -Temperature_min)+Temperature_min)+"\n")
                                grid.write("\n")
                        cmd = "../grid_thermal_map.pl Chiplet_Core"+ test_batch[0][:test_batch[0].find('_')]+".flp "+dataset_dir+"Chiplet.grid.steady "+str(length_stan)+" "+ str(length_stan)+" > "+dataset_dir+"test_Chiplet_stan"+str(epoch)+"_"+str(0)+".svg"
                
                        os.system(cmd)

                epoch_test_acc /= int(num_test/batch_size) + (num_test%batch_size > 0)
                print("Test Epoch {:05d} | Accuracy {:.4f}".format(epoch, epoch_test_acc))
                with open(dataset_dir + 'train_acc.txt','a') as Train_Acc_file:
                    Train_Acc_file.write("{:05d} {:.4f}\n".format(epoch, epoch_test_acc))
            
                  
                
            
            if epoch_test_acc < Test_Acc_min:
                Test_Acc_min = epoch_test_acc
                if os.path.exists(ckpt_dir + '/HSgcn_{}.pkl'.format(last_saved_epoch)):
                    os.remove(ckpt_dir + '/HSgcn_{}.pkl'.format(last_saved_epoch))
                #if os.path.exists(ckpt_dir + '/HSdecoder_{}.pkl'.format(last_saved_epoch)):
                #    os.remove(ckpt_dir + '/HSdecoder_{}.pkl'.format(last_saved_epoch))

                torch.save(model.state_dict(), ckpt_dir + '/HSgcn_{}.pkl'.format(epoch))
                print("model saved")
                #torch.save(node_MLP.state_dict(), ckpt_dir + '/HSdecoder_{}.pkl'.format(epoch))
                #print("decoder saved")
                
                last_saved_epoch = epoch

    else:
        model.load_state_dict(torch.load(ckpt_file, map_location=device))
        model.to(device=device)

        print('batch size: {}, num_test: {}'.format(batch_size, num_test))

        epoch_test_acc = 0
        epoch_test_MAE = 0
        epoch_test_Max = 0
        epoch_test_MaxID = 0
        epoch_test_Min = 1
        temp_maxmin = 1
        epoch_test_MinID = 0

        for i in range(0, num_test, batch_size):
            test_batch = test_data[i : min(i+batch_size, num_test)]
            g, edge_feats = read_edge(test_batch)
            node_feats1, node_feats2, node_labels = read_node(test_batch)

            g = g.to(device=device)
            node_feats1 = torch.Tensor(node_feats1).to(device=device)
            node_feats2 = torch.Tensor(node_feats2).to(device=device)
            node_labels = torch.Tensor(node_labels).to(device=device)
            edge_feats = torch.Tensor(edge_feats).to(device=device)
            
            
            acc, node_output, MAE, AEmax, AEmin = evaluate(model, g, node_feats1,node_feats2,node_labels,edge_feats)
            
            epoch_test_acc += acc
            epoch_test_MAE += MAE
            if epoch_test_Max < AEmax:
                epoch_test_Max = AEmax
                epoch_test_MaxID = i
            if temp_maxmin > AEmax:
                temp_maxmin = AEmax
                epoch_test_MinID = i
            if epoch_test_Min > AEmin:
                epoch_test_Min = AEmin
            print("Test {} | Accuracy {:.4f} | MAE {:.4f} | Max {:.4f} | Min {:.4f}".format(i, acc, MAE*(Temperature_max - Temperature_min), AEmax*(Temperature_max-Temperature_min), AEmin*(Temperature_max-Temperature_min)))
        
        epoch_test_acc /= int(num_test/batch_size)+int(num_test%batch_size>0)
        epoch_test_MAE /= int(num_test/batch_size)+int(num_test%batch_size>0)

        print("Accuracy {:.4f} | MAE {:.4f} | Max {:.4f} | Min {:.4f}".format(epoch_test_acc, epoch_test_MAE*(Temperature_max-Temperature_min), epoch_test_Max*(Temperature_max - Temperature_min), epoch_test_Min*(Temperature_max-Temperature_min)))

        test_batch = test_data[epoch_test_MaxID : min(epoch_test_MaxID+batch_size, num_test)]
        g, edge_feats = read_edge(test_batch)
        node_feats1, node_feats2, node_labels = read_node(test_batch)

        g = g.to(device=device)
        node_feats1 = torch.Tensor(node_feats1).to(device=device)
        node_feats2 = torch.Tensor(node_feats2).to(device=device)
        node_labels = torch.Tensor(node_labels).to(device=device)
        edge_feats = torch.Tensor(edge_feats).to(device=device)
            
            
        acc, node_output, MAE, AEmax, AEmin = evaluate(model, g, node_feats1,node_feats2,node_labels,edge_feats)
        
        length_pred = 64
        length_stan = 64

        with open(dataset_dir+'Chiplet.grid.steady','w') as grid:
            for m in range(length_pred):
                for n in range(length_pred):
                    grid.write(str(m*length_pred+n)+" "+ str((node_output[m*length_pred+n][0].tolist()+1)/2.0*(Temperature_max -Temperature_min)+Temperature_min)+"\n")
                grid.write("\n")
        cmd = "../grid_thermal_map.pl Chiplet_Core"+ test_batch[0][:test_batch[0].find('_')]+".flp "+dataset_dir+"Chiplet.grid.steady "+str(length_pred)+" "+ str(length_pred)+" > "+dataset_dir+"test_Chiplet_pred_max.svg"
                
        os.system(cmd)
    
        with open(dataset_dir+'Chiplet.grid.steady','w') as grid:
            for m in range(length_stan):
                for n in range(length_stan):
                    grid.write(str(m*length_stan+n)+" "+ str((node_labels[m*length_stan+n][0].tolist()+1)/2.0*(Temperature_max -Temperature_min)+Temperature_min)+"\n")
                grid.write("\n")
        cmd = "../grid_thermal_map.pl Chiplet_Core"+ test_batch[0][:test_batch[0].find('_')]+".flp "+dataset_dir+"Chiplet.grid.steady "+str(length_stan)+" "+ str(length_stan)+" > "+dataset_dir+"test_Chiplet_stan_max.svg"
                
        os.system(cmd)

        
        test_batch = test_data[epoch_test_MinID : min(epoch_test_MinID+batch_size, num_test)]
        g, edge_feats = read_edge(test_batch)
        node_feats1, node_feats2, node_labels = read_node(test_batch)

        g = g.to(device=device)
        node_feats1 = torch.Tensor(node_feats1).to(device=device)
        node_feats2 = torch.Tensor(node_feats2).to(device=device)
        node_labels = torch.Tensor(node_labels).to(device=device)
        edge_feats = torch.Tensor(edge_feats).to(device=device)
            
            
        acc, node_output, MAE, AEmax, AEmin = evaluate(model, g, node_feats1,node_feats2,node_labels,edge_feats)

        with open(dataset_dir+'Chiplet.grid.steady','w') as grid:
            for m in range(length_pred):
                for n in range(length_pred):
                    grid.write(str(m*length_pred+n)+" "+ str((node_output[m*length_pred+n][0].tolist()+1)/2.0*(Temperature_max -Temperature_min)+Temperature_min)+"\n")
                grid.write("\n")
        cmd = "../grid_thermal_map.pl Chiplet_Core"+ test_batch[0][:test_batch[0].find('_')]+".flp "+dataset_dir+"Chiplet.grid.steady "+str(length_pred)+" "+ str(length_pred)+" > "+dataset_dir+"test_Chiplet_pred_min.svg"
                
        os.system(cmd)
    
        with open(dataset_dir+'Chiplet.grid.steady','w') as grid:
            for m in range(length_stan):
                for n in range(length_stan):
                    grid.write(str(m*length_stan+n)+" "+ str((node_labels[m*length_stan+n][0].tolist()+1)/2.0*(Temperature_max -Temperature_min)+Temperature_min)+"\n")
                grid.write("\n")
        cmd = "../grid_thermal_map.pl Chiplet_Core"+ test_batch[0][:test_batch[0].find('_')]+".flp "+dataset_dir+"Chiplet.grid.steady "+str(length_stan)+" "+ str(length_stan)+" > "+dataset_dir+"test_Chiplet_stan_min.svg"
                
        os.system(cmd)


if __name__ == '__main__':
    main()       


