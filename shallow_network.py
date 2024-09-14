from math import *
import torch, gzip
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier
import seaborn as sns

class ShallowModel(nn.Module):

    def __init__(self, n_neurons = 128):
        super(ShallowModel, self).__init__()
        self.fc1 = nn.Linear(784, n_neurons)
        self.fc2 = nn.Linear(n_neurons, 10)
        
        torch.nn.init.uniform_(self.fc1.weight, -0.001, 0.001)
        torch.nn.init.uniform_(self.fc1.bias,  -0.001, 0.001)
        torch.nn.init.uniform_(self.fc2.weight, -0.001, 0.001)
        torch.nn.init.uniform_(self.fc2.bias,  -0.001, 0.001)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

# On extrait le jeu d'apprentissage et le jeu de test 
((data_train,label_train),(data_test,label_test)) = torch.load(gzip.open('mnist.pkl.gz'))
data_train_np = data_train.numpy()
label_train_np = label_train.numpy()
dataset = torch.utils.data.TensorDataset(data_train[:floor((data_train.size(0) * 0.1))],label_train[:floor(data_train.size(0) * 0.1)])
validation_dataset = torch.utils.data.TensorDataset(data_train[floor(data_train.size(0) * 0.9):], label_train[floor(label_train.size(0) * 0.9):])
test_dataset = torch.utils.data.TensorDataset(data_test, label_test)

# On crée le model 
model = NeuralNetClassifier(
    ShallowModel,
    criterion=nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    max_epochs=3,
    batch_size=5,
    lr=0.001,
    verbose=False,
    train_split=None
)

# HYPERPARAMETRE ----------------------------------------------------
batch_size = 5
nb_epochs = 10
eta = 0.01
w_min = -0.001
w_max = 0.01
nb_neuron = 128

param_grid = {
    'batch_size': [5, 10],
    'max_epochs': [3, 5, 10],
    #'module__n_neurons':[128, 256]
    #'nb_neuron': [128, 256, 784],
    #'optimizer__lr': [0.01, 0.001]
}

# -------------------------------------------------------------------

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(data_train, np.argmax(label_train, axis=1))


# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
# # Définition de la fonction de perte et l'optimiseur
# loss_func = torch.nn.CrossEntropyLoss()
# optim = torch.optim.Adam(model.parameters(), lr=eta)



# loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
# validation_loader = torch.utils.data.DataLoader(validation_dataset, 1, shuffle=False)
# test_loader = torch.utils.data.DataLoader(test_dataset, 1, shuffle=False)

# # Affichage des données 
# yTrain = []
# yPred = []

# for n in range(nb_epochs):
#     for x,t in loader:    
#         y = model.forward(x)
#         loss = loss_func(y, t)
        
#         optim.zero_grad()
#         loss.backward()
#         optim.step()
    
#     for x,t in validation_loader:
#         y = model.forward(x)
        
#     acc = 0
#     nb_data = 0
#     for x, t in test_loader:
#         y = model.forward(x)
#         yTrain.append(torch.argmax(y,1))
#         yPred.append(torch.argmax(t,1))
#         nb_data += 1
#         acc += torch.argmax(y,1) == torch.argmax(t,1)
        
#     plt.figure(figsize=(6,4))
#     matrix = confusion_matrix(np.array(yTrain), np.array(yPred))
#     sns.heatmap(matrix, annot=True, fmt='d', cmap="viridis")
#     plt.ylabel("Expected")
#     plt.xlabel("Output")
#     plt.show()
#     print(acc/nb_data)
        
