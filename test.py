import numpy as np
from alexnet_3ft import AlexNet
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision
device = torch.device("cuda:0")


WIDTH = 200
HEIGHT = 150
LR = 0.0003
EPOCHS = 1

#Tensorboard
writer = SummaryWriter('./logs')


model = AlexNet()
model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

all_data = np.load('final_training_data.npy', allow_pickle=True)

inputs = all_data[:,0]
labels = all_data[:,1]

inputs_tensors = torch.stack([torch.FloatTensor(i) for i in inputs])
labels_tensors = torch.stack([torch.LongTensor(i) for i in labels])


data_set = torch.utils.data.TensorDataset(inputs_tensors,labels_tensors)
data_loader = torch.utils.data.DataLoader(data_set, batch_size=1,shuffle=True, num_workers=1)

if __name__ == '__main__':
	for epoch in range(EPOCHS):
		print('Number of epoch:			 ',epoch)
		running_loss = 0.0
		for i,data in enumerate(data_loader , 0):
			inputs= data[0].to(device)
			labels= data[1].to(device)
			optimizer.zero_grad()
			#set_trace()
			inputs = torch.unsqueeze(inputs, 1)

			outputs = model(inputs)
			loss = criterion(outputs , torch.max(labels, 1)[1])
			loss.backward()
			optimizer.step()

			running_loss +=loss.item()


			if i % 2000 == 1999:    # print every 2000 mini-batches
				av_loss = running_loss/2000
				writer.add_scalar('Train/Loss', av_loss, epoch, i)				
				print('[%d, %5d] loss: %.3f' %
	                  (epoch + 1, i + 1, running_loss / 2000))
				running_loss = 0.0

		MODEL_NAME = 'pygta5-{}-{}-{}-epochs.model'.format(LR, 'alexnet', epoch)		
		torch.save(model.state_dict(), MODEL_NAME)
	print('finished')
     

writer.close()
MODEL_NAME = 'pygta5-{}-{}-{}-epochs.model'.format(LR, 'alexnet', EPOCHS)
torch.save(model.state_dict(), MODEL_NAME)