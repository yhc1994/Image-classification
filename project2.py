import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.00005, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
#initialize the val_loss and train_loss as a empty matrix
val_loss = []
train_loss = []

for epoch in range(8):  # loop over the trainset（2700 each epoch） multiple times
    running_train_loss = 0.0
    running_val_loss = 0.0
    scheduler.step()
    for i, data in enumerate(trainloader , 0):
        # get the inputs
        images, labels = data
        inputs, labels = images.cuda(),labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_train_loss += loss.item()
        if i % 500 == 499:    # print 1 epoch
             print('[%d, %5d] train_loss: %.3f' %
                  (epoch + 1, i + 1, running_train_loss / 500))
             train_loss.append(running_train_loss/500)
                
    for i, data in enumerate(testloader, 0):
        # get the inputs
        images, labels = data
        inputs, labels = images.cuda(),labels.cuda()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        # print statistics
        running_val_loss += loss.item()
        if i % 88 == 87:    # print 1 epoch
             print('[%d, %5d] val_loss: %.3f' %
                  (epoch + 1, i + 1, running_val_loss /88))
             val_loss.append(running_val_loss/88)
             running_train_loss = 0.0
             running_val_loss = 0.0
 
                
# functions to show the loss
t = np.arange(8)  

plt.plot(t+1, train_loss,color='red',linestyle='--')
plt.plot(t+1, val_loss,color='blue',linestyle='-.')

plt.title('loss curve')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()