import torch, gzip
import torch.nn as nn
import torch.nn.functional as F

class ShallowModel(nn.Module):

    def __init__(self, nb_neuron):
        super(ShallowModel, self).__init__()
        self.fc1 = nn.Linear(784, nb_neuron)
        self.fc2 = nn.Linear(nb_neuron, 10)
        
        torch.nn.init.uniform_(self.fc1.weight, -0.001, 0.001)
        torch.nn.init.uniform_(self.fc2.weight, -0.001, 0.001)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
    
batch_size = 1
nb_epochs = 1
eta = 0.00001
w_min = -0.001
w_max = 0.001

model = ShallowModel(784)
# torch.nn.init.uniform_(model.weight,w_min,w_max)
loss_func = torch.nn.CrossEntropyLoss(reduction='sum')
optim = torch.optim.Adam(model.parameters(), lr=eta)

((data_train,label_train),(data_test,label_test)) = torch.load(gzip.open('mnist.pkl.gz'))	
dataset = torch.utils.data.TensorDataset(data_train,label_train)
test_dataset = torch.utils.data.TensorDataset(data_test, label_test)
loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, 1, shuffle=False)

for n in range(nb_epochs):
    for x,t in loader:    
        y = model.forward(x)
        loss = loss_func(t,y)
        loss.backward()
        optim.step()
        optim.zero_grad()
    
    acc = 0
    nb_data = 0
    for x, t in test_loader:
        y = model.forward(x)
        print(acc)
        nb_data += 1
        acc += torch.argmax(y,1) == torch.argmax(t,1)
        
    
    print(acc/nb_data)
        
