import numpy as np
from alexnet_3ft import AlexNet
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from IPython.core.debugger import set_trace

device = torch.device("cuda:0")


WIDTH = 200
HEIGHT = 150
LR = 0.0003
EPOCHS = 30

#Tensorboard
#writer = SummaryWriter()


model = AlexNet()
model.to(device)
#print('If is cuda: ', next(model.parameters()).is_cuda)			#Działa


criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

all_data = np.load('final_training_data.npy', allow_pickle=True)

inputs = all_data[:,0]
labels = all_data[:,1]

#inputs.to(device)
#labels.to(device)

inputs_tensors = torch.stack([torch.FloatTensor(i) for i in inputs])
labels_tensors = torch.stack([torch.LongTensor(i) for i in labels])
#inputs_tensors = torch.stack([torch.from_numpy(i).float().to(device) for i in inputs])
#labels_tensors = torch.stack([torch.from_numpy(i).float().to(device) for i in labels])

#inputs_tensors = torch.stack([torch.cuda.FloatTensor(i) for i in inputs])
#labels_tensors = torch.stack([torch.cuda.FloatTensor(i) for i in labels])

#inputs_tensors.to(device)
#labels_tensors.to(device)

data_set = torch.utils.data.TensorDataset(inputs_tensors,labels_tensors)
data_loader = torch.utils.data.DataLoader(data_set, batch_size=512,shuffle=True, num_workers=3)

if __name__ == '__main__':
	for epoch in range(EPOCHS):
		print('Number of epoch:			 ',epoch)
		running_loss = 0.0
		for i,data in enumerate(data_loader , 0):
			#print(i)
			inputs= data[0].to(device)
			#inputs = torch.cuda.FloatTensor(inputs)
			labels= data[1].to(device)
			#labels = torch.cuda.FloatTensor(labels)

			optimizer.zero_grad()
			#set_trace()
			inputs = torch.unsqueeze(inputs, 1)

			#inputs.to(device)
			#labels.to(device)

			outputs = model(inputs)
			loss = criterion(outputs , torch.max(labels, 1)[1])
			loss.backward()
			optimizer.step()

			running_loss +=loss.item()

			if i % 2000 == 1999:    # print every 2000 mini-batches
				print('[%d, %5d] loss: %.3f' %
	                  (epoch + 1, i + 1, running_loss / 2000))
				running_loss = 0.0

				
		if epoch %5 == 4:
			MODEL_NAME = 'pygta5-{}-{}-{}-epochs.model'.format(LR, 'alexnet', epoch)
			torch.save(model.state_dict(), MODEL_NAME)
	print('finished')
     


MODEL_NAME = 'pygta5-{}-{}-{}-epochs.model'.format(LR, 'alexnet', EPOCHS)
torch.save(model.state_dict(), MODEL_NAME)
