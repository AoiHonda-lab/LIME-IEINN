import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.special import comb
import time
from . import narray_op as n_op
from . import preprocessing_layer
from . import output_layer
from itertools import combinations
import numpy as np
import csv

class IE(nn.Module):
    def __init__(self, dataloader,  additivity_order=None, narray_op='Algebraic', preprocessing='PreprocessingLayerPercentile'):
        super(IE,self).__init__()
        self.X_data = dataloader.dataset[:][0]
        self.additivity_order =  additivity_order   # 何加法までtノル?��?を計算するか
        self.narray_op = narray_op                  # tノル?��?の種?��?
        self.preprocessing = preprocessing          # 前�??��?��?の種?��?
        columns_num = self.X_data.size()[1]         #説明変数の数
        self.columns_num = columns_num
        self.set_list = (self.conb(columns_num, self. additivity_order))[1:] # ?��?合�??��リス?��?
                    
        if self. additivity_order == None:
            self. additivity_order = columns_num

        if self. additivity_order > columns_num:
            raise IndexError('" additivity_order" must be less than the "number of features"')
        if self.narray_op not in ['Algebraic', 'Min']:#, 'Lukasiewicz', 'Dubois', 'Hamacher']:
            raise ValueError('narray_op / Algebraic, Min')#, Lukasiewicz, Dubois, Hamacher')
        if self.preprocessing not in ['PreprocessingLayerPercentile', 'PreprocessingLayerStandardDeviation', 'PreprocessingLayerMaxMin', 'PreprocessingLayerNone']:
            raise ValueError('preprocessing / PreprocessingLayerPercentile, PreprocessingLayerStandardDeviation, PreprocessingLayerMaxMin, PreprocessingLayerNone')
    
        # PreprocessingLayer
        preprocessing_cls = getattr(preprocessing_layer, self.preprocessing)
        for i in range(0,columns_num):
            exec("self.preprocessing" + str(i+1) + "= preprocessing_cls(dataloader.dataset, "+str(i)+")")
        #IEILayer
        iei_cls = getattr(n_op, self.narray_op)
        self.iei = iei_cls(self. additivity_order)
        # OutputLayer
        output_nodesSum = 0
        for i in range(1,self. additivity_order+1):
            output_nodesSum += comb(columns_num, i, exact=True)
        self.output = output_layer.OutputLayer(columns_num, output_nodesSum)
            
    def forward(self, x):
        #device = 'cuda' if torch.cuda.is_available() else 'cpu'
        columns_num = x.size()[1]
        self.nodes=torch.tensor((), dtype=torch.float)

        #preprocessing
        for i in range(0, columns_num):
            exec("x" + str(i+1) + "= self.preprocessing" + str(i+1) + "(x[:," + str(i) + "].view(x.size()[0],1))")
            if self.preprocessing != 'PreprocessingLayerNone':
                exec("x_sig" + str(i+1) + "= torch.sigmoid(x" + str(i+1) + ")")
            else:
                exec("x_sig" + str(i+1) + "= x" + str(i+1))
            exec("self.nodes=torch.cat((self.nodes,x_sig" + str(i+1) + "),dim=1)")
        #iei
        hidden=self.iei(self.nodes)
        #output
        return self.output(hidden)

    def fit_and_valid(self, train_Loader, test_Loader, criterion, optimizer, scheduler, early_stopping, device='cpu', epochs=100, regularization='none', lmd=0.2, ):
        start = time.time()
        self.train_loss_list = []
        self.val_loss_list = []
        self.lrs_list  = []
        self.r2_list = []
        self.not_mono_num_list = []
        self.not_mono_num_percentage = []
        self.f_list=[]
        f = 0 #何点?��?合まで単調性条件を満たして?��?るか調べるた�??��??��フラグ
    
        for epoch in range(epochs):
            self.train_loss = 0
            self.val_loss = 0
            
            #train
            self.train()
            for i, (images, labels, sample_weights) in enumerate(train_Loader):
                images, labels = images.to(device), labels.to(device)
                labels = labels.reshape(-1, 1)

                # Zero your gradients for every batch
                optimizer.zero_grad()
                # Make predictions for this batch
                outputs = self(images)
                
                reg_sum = torch.tensor(0., requires_grad=True, dtype=torch.float)               #正?��?化�??の値
                self.d_w = dict(zip(self.set_list, self.output.weight[0])) # 辞書化した重み
                
                if regularization == 'none':
                    pass 
                                
                elif regularization == 'mono_reguralization':
                    # 単調性の判別式から、単調性を満たさな?��?部?��?の重みの溢れた?��?の【Min】だけ足し合わせる正?��??��?
                    for A in self.set_list:
                        i_sum = torch.tensor(0., requires_grad=True, dtype=torch.float)
                        for i in A:
                            sum = 0
                            for B in self.set_list:
                                # Bは�??��?iを含む かつ Aの真部?��??��??��?
                                if (set({i})<=(set(B))) and (set(B)<(set(A))):
                                    sum = sum - self.d_w[B]

                            # 単調性条件を満たしてな?��?場合�??��??��?Aと?��??��?Bの差?��?をリストに追?��?
                            if self.d_w[A] < (sum):
                                if i_sum == 0:
                                    i_sum = torch.norm(self.d_w[A]-sum)
                                elif i_sum < torch.norm(self.d_w[A]-sum):
                                    i_sum = torch.norm(self.d_w[A]-sum)
                                        
                        reg_sum = reg_sum + torch.norm(i_sum)
    
                # Compute the loss
                loss = criterion(outputs, labels) #純粋なloss記録用
                self.train_loss += loss.item()*len(labels)
                
                loss = torch.dot(sample_weights.reshape(-1), ((outputs - labels).reshape(-1))**2) / labels.shape[0] + lmd * reg_sum
                
                # Compute the its gradients
                loss.backward()
                # Adjust learning weights
                optimizer.step()
                self.lrs=optimizer.param_groups[0]["lr"]
            
            self.lrs_list.append(optimizer.param_groups[0]["lr"])
            scheduler.step(self.train_loss)
            avg_train_loss = self.train_loss / len(train_Loader.dataset[:][:2])

            #val
            self.eval()
            with torch.no_grad():
                for images, labels, sample_weights in test_Loader:
                    images = images.to(device)
                    labels = labels.to(device).reshape(-1, 1)

                    outputs = self(images)
                    loss = torch.dot(sample_weights.reshape(-1), ((outputs - labels).reshape(-1))**2) / labels.shape[0] + lmd * reg_sum
                    self.val_loss += loss.item()*len(labels)
            avg_val_loss = self.val_loss / len(test_Loader.dataset[:][:2])
            early_stopping(avg_val_loss, self)
            self.r2_list.append(self.r2_score(test_Loader))
            self.not_mono_num_list.append(len(self.test_monotone()))
            self.not_mono_num_percentage.append(len(self.test_monotone())/len(self.set_list))

            print ('Epoch [{}/{}], loss: {loss:.8f} val_loss: {val_loss:.8f}' 
                        .format(epoch+1, epochs, i+1, loss=avg_train_loss, val_loss=avg_val_loss))
            self.train_loss_list.append(avg_train_loss)
            self.val_loss_list.append(avg_val_loss)
            
            if early_stopping.early_stop:
                break

        print("time")
        self.preprocessing_time = time.time() - start
        print(self.preprocessing_time)

        return self.val_loss

    def predict(self, x):
        outputs = self(x)
        return outputs

    def plot_train(self):
        plt.figure(figsize=(8,6))
        plt.plot(self.train_loss_list,label='train data', lw=3, c='b')
        plt.xticks(size=14)
        plt.yticks(size=14)
        plt.grid(lw=2)
        plt.legend(fontsize=14)
        plt.show()
        return 0

    def plot_test(self):
        plt.figure(figsize=(8,6))
        plt.plot(self.val_loss_list,label='test data', lw=3, c='b')
        plt.xticks(size=14)
        plt.yticks(size=14)
        plt.grid(lw=2)
        plt.legend(fontsize=14)
        plt.show()
        return 0
        
    def plot(self):
        plt.figure(figsize=(8,6))
        plt.plot(self.train_loss_list,label='train data', lw=3, c='b')
        plt.plot(self.val_loss_list,label='test data', lw=3, c='r')
        plt.xticks(size=14)
        plt.yticks(size=14)
        plt.grid(lw=2)
        plt.legend(fontsize=14)
        plt.show()
        return 0
    
    def r2_score(self,test_Loader):
        test_mean=torch.mean(test_Loader.dataset[:][1])
        test_mean_list=torch.ones(len(test_Loader.dataset[:][1]),1, dtype=torch.float)*test_mean
        mse_loss = torch.nn.MSELoss()
        mean_loss=mse_loss(test_Loader.dataset[:][1],test_mean_list)*len(test_Loader.dataset[:][1])
        return 1-self.val_loss/mean_loss.item()

    #　?��?み合わせを作る関数。�?)conb(3,2)としたら[(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3)]
    def conb(self, ie_data_len,  additivity_order):
        items = [i for i in range(1, ie_data_len+1)]
        subsets=[]
        for i in range(len(items) + 1):
            if i >  additivity_order:#何加法的まで?��?
                break
            for c in combinations(items, i):
                subsets.append(c)
        hh = subsets
        return hh #?��?合わせ�??��?��?
    
    def test_monotone(self): #単調性を満たして?��?るか判別する関数
        ng_list = [] # ファジィ測度が単調性を満たして?��?な?��??��?合�??��リス?��?
        self.d_w_np = dict(zip(self.set_list, self.get_weight()))

        for A in self.set_list:
            for i in A:
                sum = 0
                for B in self.set_list:
                    # Bは�??��?iを含む かつ Aの真部?��??��??��?
                    if (set({i})<=(set(B))) and (set(B)<(set(A))):
                        sum = sum - self.d_w_np[B]

                # 単調性条件を満たしてな?��?場合リストに追?��?して抜け?��?
                if self.d_w_np[A] < (sum):
                    ng_list.append(A)
                    break
        return ng_list
    
    def get_weight(self):
        np_weight = self.output.weight.to('cpu').detach().numpy().copy()
        return np.ravel(np_weight)
    
    
    
    #　?��?み合わせを作る関数
    def power_sets(self):
        items = [i for i in range(1, self.columns_num+1)]
        sub_sets=[]
        for i in range(len(items) + 1):
            if i > self.additivity_order:
                break
            for c in combinations(items, i):
                sub_sets.append(c)
        all_sets = sub_sets
        return all_sets

    # メビウス変換・?��?変換を行う関数
    def mobius(self, mobius_transformation=False):
        fuzzy_ = []
        all_sets = (self.power_sets()) 
        self.np_houjo_w = (self.output.weight).to('cpu').detach().numpy().copy() #最後�??��層の重みをnumpyに変換
        l_data = self.np_houjo_w[0].tolist()
        l_data.insert(0, 0)
        data = np.array(l_data)

        #メビウス変換
        if mobius_transformation==True:
            for i in range(0, len(data)):
                f = 0
                for j in range(0, len(data)):
                    A = all_sets[i]
                    B = all_sets[j]
                    if  set(B) <= set(A): #"<="は部?��??��?合か?��?"<"にしたら真部?��??��?合か判定できる
                        f += data[j]*(-1)**(len(all_sets[i])-len(all_sets[j]))
                fuzzy_.append(f)
            d_fuzzy = dict(zip(all_sets, fuzzy_))

        #メビウス?��?変換
        else:
            for i in range(0, len(data)):
                f = 0
                for j in range(0, len(data)):
                    A = all_sets[i]
                    B = all_sets[j]
                    if  set(B) <= set(A): #"<="は部?��??��?合か?��?"<"にしたら真部?��??��?合か判定できる
                        f += data[j]
                fuzzy_.append(f)
            d_fuzzy = dict(zip(all_sets, fuzzy_))

        return d_fuzzy #部?��??��?合�??��?��?み合わせと、そのファジィ測度の値を辞書の形にしたも�??��を返す

    #　シャプレイ値を返す関数
    def shapley(self, value_type='mobius'):
        all_sets = (self.power_sets()) 
        self.np_houjo_w = (self.output.weight).to('cpu').detach().numpy().copy() #最後�??��層の重みをnumpyに変換

        #メビウス変換値でなければメビウス変換
        if value_type=='fuzzy':
            d_fuzzy = self.mobius(mobius_transformation=True)
        #メビウス変換値ならそのまま
        if value_type=='mobius':
            l_data = self.np_houjo_w[0].tolist()
            l_data.insert(0, 0)
            data = np.array(l_data)
            d_fuzzy = dict(zip(all_sets, data))

        shapley = []
        for i in range(1, self.columns_num+1):
            shapley_val = 0
            A = all_sets[i]
            for B in all_sets:
                if set(A) <= set(B):
                    shapley_val += d_fuzzy[B] / len(B)
            shapley.append(shapley_val) 
        shapley = dict(zip(all_sets[1:], shapley))
        return shapley

    #　Owen Valueを計算する関数
    def owen(self, group, value_type='mobius'):
        all_sets = (self.power_sets()) 
        self.np_houjo_w = (self.output.weight).to('cpu').detach().numpy().copy() #最後�??��層の重みをnumpyに変換

        #メビウス変換値でなければメビウス変換
        if value_type=='fuzzy':
            d_fuzzy = self.mobius(mobius_transformation=True)  
        #メビウス変換値ならそのまま
        if value_type=='mobius':
            l_data = self.np_houjo_w[0].tolist()
            l_data.insert(0, 0)
            data = np.array(l_data)
            d_fuzzy = dict(zip(all_sets, np.ravel(data))) # 辞書化した重み

        owen = []          # Owen Value
        g_num = len(group) # グループ数

        for i in all_sets[1:self.columns_num+1]:
            owen_val = 0
            for S in all_sets:
                count = np.zeros(g_num) # ?��?グループに含まれる�??��?数
                i_group = 0 # iが何番目のグループに属するか格納す?��?
                j = 0
                if set(i) <= set(S):
                    for g in group:
                        common  =  set(g) & set(S) # 和集?��?
                        count[j] = len(common)
                        if set(common) >= set(i):
                            i_group = j
                        else:
                            pass
                        j += 1

                    # np.count_nonzero(count)はメンバ�??��がゼロではな?��?グループ数。つまりSのメンバ�??��が属するグループ数
                    # count[i_group]はiが属するグループに属する�??��?数。iも込み?��?
                    owen_val += d_fuzzy[S] / (np.count_nonzero(count) * count[i_group])
                else:
                    pass
            owen.append(owen_val)
        owen = dict(zip(all_sets[1:], owen))
        return owen
    
    #出力層　重み　エクスポ�??��?��?
    def weight_exp(self):
        with open('output_weight.csv', 'w',newline='') as f:
            writer = csv.writer(f)
            for weight in self.output.weight:
                for i in weight:
                    writer.writerow([i.tolist()])
    
    #全パラメータ　エクスポ�??��?��?
    def param_exp(self):
        with open('param.csv', 'w',newline='') as f:
            writer = csv.writer(f)
            for p in self.parameters():
                for i in p:
                    if i.ndim == 1:
                        writer.writerow(i.tolist())
                    else:
                        writer.writerow([i.tolist()])