import numpy as np
from numpy import genfromtxt
import random

Num_layout = 400
Num_power = 20

num_train = int(Num_layout*0.85)
num_test = Num_layout-num_train


node_feats = genfromtxt('./data/Power_{}_{}.csv'.format(0,0), delimiter=',')
node_labels = genfromtxt('./data/Temperature_{}_{}.csv'.format(0,0), delimiter=',')

edge_feats = genfromtxt('./data/Edge_{}_{}.csv'.format(0, 0), delimiter=',')

Power_sum = 0
for m in range(node_feats.shape[0]):
    if m > 12291:
        node_feats[m][1] = 0.1
    Power_sum = Power_sum + node_feats[m][1]


for m in range(edge_feats.shape[0]):
    if edge_feats[m][0] in range(12288, 12300) and edge_feats[m][1] in range(12288, 12300):
        edge_feats[m][2] = 0.1

tPower_max = Power_sum
tPower_min = Power_sum
Power_max = np.amax(node_feats[:, 1])
Power_min = np.amin(node_feats[:, 1])
Temperature_max = np.amax(node_labels[:, 1])
Temperature_min = np.amin(node_labels[:, 1])
Conductance_max = np.amax(edge_feats[:, 2])
Conductance_min = np.amin(edge_feats[:, 2])



for i in range(Num_layout):
    for j in range(Num_power):
        node_feats = genfromtxt('./data/Power_{}_{}.csv'.format(i,j), delimiter=',')
        node_labels = genfromtxt('./data/Temperature_{}_{}.csv'.format(i,j), delimiter=',')

        edge_feats = genfromtxt('./data/Edge_{}_{}.csv'.format(i,j), delimiter=',')

        Power = open('./newdata/Power_{}_{}.csv'.format(i, j), 'w')
        totalPower = open('./newdata/totalPower_{}_{}.csv'.format(i,j),'w')
        Power_sum = 0
        for m in range(node_feats.shape[0]):
            if m > 12291:
                node_feats[m][1] = 0.1
            Power.write(str(m) + "," + str(node_feats[m][1]) + "\n")
            Power_sum = Power_sum + node_feats[m][1]
        for m in range(node_feats.shape[0]):
            totalPower.write(str(Power_sum)+"\n")
        
        if tPower_max < Power_sum:
            tPower_max = Power_sum

        if tPower_min > Power_sum:
            tPower_min = Power_sum

        totalPower.close()


        Power.close()
    
        Temperature =  open('./newdata/Temperature_{}_{}.csv'.format(i, j), 'w')
        
        for m in range(node_labels.shape[0]):
            Temperature.write(str(m) + "," + str(node_labels[m][1]) + "\n")
        Temperature.close()
            
        Edge = open('./newdata/Edge_{}_{}.csv'.format(i, j), 'w')
        for m in range(edge_feats.shape[0]):
            if edge_feats[m][0] in range(12288, 12300) and edge_feats[m][1] in range(12288, 12300):
                edge_feats[m][2] = 0.1        
            Edge.write(str(edge_feats[m][0].astype(int)) + "," + str(edge_feats[m][1].astype(int)) + "," + str(edge_feats[m][2]) + "\n")
            
        Edge.close()
	


	
	

        PowerTemp1 = np.amax(node_feats[:, 1])
        if PowerTemp1 > Power_max:
            Power_max = PowerTemp1
        
        PowerTemp2 = np.amin(node_feats[:, 1])
        if PowerTemp2 < Power_min:
            Power_min = PowerTemp2

        TemperatureTemp1 = np.amax(node_labels[:, 1])
        if TemperatureTemp1 > Temperature_max:
            Temperature_max = TemperatureTemp1
        
        TemperatureTemp2 = np.amin(node_labels[:, 1])
        if TemperatureTemp2 < Temperature_min:
            Temperature_min = TemperatureTemp2

        ConductanceTemp1 = np.amax(edge_feats[:, 2])
        if ConductanceTemp1 > Conductance_max:
            Conductance_max = ConductanceTemp1
        
        ConductanceTemp2 = np.amin(edge_feats[:, 2])
        if ConductanceTemp2 < Conductance_min:
            Conductance_min = ConductanceTemp2
        
        print(i,tPower_max, tPower_min, Power_max,Power_min,Temperature_max,Temperature_min,Conductance_max,Conductance_min)



	
with open('./data/MaxMinValues.csv', 'w') as MaxMinFile:
            MaxMinFile.write("tPower_max, tPower_min, Power_max,Power_min,Temperature_max,Temperature_min,Conductance_max,Conductance_min\n")
            MaxMinFile.write(str(tPower_max)+","+str(tPower_min)+","+str(Power_max) + "," + str(Power_min) + "," + str(Temperature_max) + "," + str(Temperature_min) + "," + str(Conductance_max) + "," + str(Conductance_min) + "\n")

with open('./newdata/MaxMinValues.csv', 'w') as MaxMinFile:
            MaxMinFile.write("tPower_max, tPower_min, Power_max,Power_min,Temperature_max,Temperature_min,Conductance_max,Conductance_min\n")
            MaxMinFile.write(str(tPower_max)+","+str(tPower_min)+","+str(Power_max) + "," + str(Power_min) + "," + str(Temperature_max) + "," + str(Temperature_min) + "," + str(Conductance_max) + "," + str(Conductance_min) + "\n")

test = []
while len(test) < num_test:
    ID = random.randint(0, Num_layout-1)
    if ID not in test:
        test.append(ID)

    print(len(test),ID)


testFile = open('./data/test_data.csv', 'w')
trainFile = open('./data/train_data.csv', 'w')

for i in range(Num_layout):
    if i in test:
        for j in range(Num_power):
            testFile.write(str(i)+"_"+str(j)+"\n")
    else:
        for j in range(Num_power):
            trainFile.write(str(i)+"_"+str(j)+"\n")
        
testFile.close()
trainFile.close()

testFile = open('./newdata/test_data.csv', 'w')
trainFile = open('./newdata/train_data.csv', 'w')

for i in range(Num_layout):
    if i in test:
        for j in range(Num_power):
            testFile.write(str(i)+"_"+str(j)+"\n")
    else:
        for j in range(Num_power):
            trainFile.write(str(i)+"_"+str(j)+"\n")
        
testFile.close()
trainFile.close()        
