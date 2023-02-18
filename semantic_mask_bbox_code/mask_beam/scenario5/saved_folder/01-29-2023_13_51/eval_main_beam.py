'''
Main script for training and testing a DL model (resnet18) for mmWave beam prediction
@author: Gouranga
'''

def main():
    
    import os
    import datetime
    import shutil
    
    import torch as t
    import torch.cuda as cuda
    import torch.optim as optimizer
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import torchvision.transforms as transf
    from torchsummary import summary
    
    import numpy as np
    import pandas as pd
    
    from data_feed import DataFeed


    
    ############################################
    ########### Create save directory ##########
    ############################################
    
    # year month day 
    dayTime = datetime.datetime.now().strftime('%m-%d-%Y')
    # Minutes and seconds 
    hourTime = datetime.datetime.now().strftime('%H_%M')
    print(dayTime + '\n' + hourTime)
    
    pwd = os.getcwd() + '//' + 'saved_folder' + '//' + dayTime + '_' + hourTime 
    print(pwd)
    # Determine whether the folder already exists
    isExists = os.path.exists(pwd)
    if not isExists:
        os.makedirs(pwd)    
        
    
    #copy the training files to the saved directory
    shutil.copy('./main_beam.py', pwd)
    shutil.copy('./data_feed.py', pwd)
    shutil.copy('./scenario5_train_mask_single_sample.csv', pwd)
    shutil.copy('./scenario5_val_mask_single_sample.csv', pwd)
    shutil.copy('./scenario5_test_mask_single_sample.csv', pwd)

    
    #create folder to save analysis files and checkpoint
    
    save_directory = pwd + '//' + 'saved_analysis_files'
    checkpoint_directory = pwd + '//' + 'checkpoint'

    isExists = os.path.exists(save_directory)
    if not isExists:
        os.makedirs(save_directory) 
        
    isExists = os.path.exists(checkpoint_directory)
    if not isExists:
        os.makedirs(checkpoint_directory)         
    
    ############################################    
    
    ########################################################################
    ######################### Hyperparameters ##############################
    ########################################################################
    
    batch_size = 64
    val_batch_size = 1
    lr = 1e-3
    decay = 1e-4
    num_epochs = 30
    train_size = [1]
    
    ########################################################################
    ########################### Data pre-processing ########################
    ########################################################################
    
    img_resize = transf.Resize((32, 32))
    img_norm = transf.Normalize(mean = (0.1307,), std = (0.3081,))
    proc_pipe = transf.Compose(
        [transf.ToPILImage(),
         img_resize,
         transf.ToTensor(),
         img_norm]
    )
    
    
    
    train_dir = 'scenario5_train_mask_single_sample.csv'
    val_dir = 'scenario5_val_mask_single_sample.csv'
    train_loader = DataLoader(DataFeed(train_dir, transform=proc_pipe),
                              batch_size=batch_size,
                              #num_workers=8,
                              shuffle=False)
    val_loader = DataLoader(DataFeed(val_dir, transform=proc_pipe),
                            batch_size=val_batch_size,
                            #num_workers=8,
                            shuffle=False)


    #Defining the convolutional neural network
    class LeNet5(nn.Module):
        def __init__(self, num_classes):
            super(LeNet5, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
                nn.BatchNorm2d(6),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2, stride = 2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2, stride = 2))
            self.fc = nn.Linear(400, 120)
            self.relu = nn.ReLU()
            self.fc1 = nn.Linear(120, 84)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(84, num_classes)
            
        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc(out)
            out = self.relu(out)
            out = self.fc1(out)
            out = self.relu1(out)
            out = self.fc2(out)
            return out

    with cuda.device(0):
       
        acc_loss = 0
        itr = []
        for idx, n in enumerate(train_size):
            print('```````````````````````````````````````````````````````')
            # print('Training size is {}'.format(n))
            # Build the network:
            net = LeNet5(num_classes=65)
            net = net.cuda()
            summary(net.cuda(), (1,32, 32))
    
            #  Optimization parameters:
            criterion = nn.CrossEntropyLoss()
            opt = optimizer.Adam(net.parameters(), lr=lr, weight_decay=decay)
            LR_sch = optimizer.lr_scheduler.MultiStepLR(opt, [10,20], gamma=0.1, last_epoch=-1)
    
            count = 0
            running_loss = []
            running_top1_acc = []
            running_top2_acc = []
            running_top3_acc = []
            running_top5_acc = []
            
            best_accuracy = 0
            
            
    ########################################################################
    ########################################################################
    ################### Load the model checkpoint ##########################    
    test_dir = './scenario5_test_mask_single_sample.csv'
    checkpoint_path = './checkpoint/LeNet5_64_beam'   
    net.load_state_dict(t.load(checkpoint_path))
    net.eval() 
    net = net.cuda()   
    
    test_loader = DataLoader(DataFeed(test_dir, transform=proc_pipe),
                            batch_size=val_batch_size,
                            #num_workers=8,
                            shuffle=False) 
    
    print('Start validation')
    ave_top1_acc = 0
    ave_top2_acc = 0
    ave_top3_acc = 0
    ave_top5_acc = 0
    ind_ten = t.as_tensor([0, 1, 2, 3, 4], device='cuda:0')
    top1_pred_out = []
    top2_pred_out = []
    top3_pred_out = []
    top5_pred_out = []
    gt_beam = []
    total_count = 0
    for val_count, (imgs, labels) in enumerate(test_loader):
        net.eval()
        x = imgs.cuda()
        opt.zero_grad()
        labels = labels.cuda()
        total_count += labels.size(0)
        out = net.forward(x)
        _, top_1_pred = t.max(out, dim=1)
        
        gt_beam.append(labels.detach().cpu().numpy()[0])
        
        top1_pred_out.append(top_1_pred.detach().cpu().numpy()[0])
        sorted_out = t.argsort(out, dim=1, descending=True)
        
        top_2_pred = t.index_select(sorted_out, dim=1, index=ind_ten[0:2])
        top2_pred_out.append(top_2_pred.detach().cpu().numpy()[0])
        
        top_3_pred = t.index_select(sorted_out, dim=1, index=ind_ten[0:3])
        top3_pred_out.append(top_3_pred.detach().cpu().numpy()[0])
        
        top_5_pred = t.index_select(sorted_out, dim=1, index=ind_ten)
        top5_pred_out.append(top_5_pred.detach().cpu().numpy()[0])                      
        
        reshaped_labels = labels.reshape((labels.shape[0], 1))
        tiled_2_labels = reshaped_labels.repeat(1, 2)
        tiled_3_labels = reshaped_labels.repeat(1, 3)
        tiled_5_labels = reshaped_labels.repeat(1, 5) 
        
        batch_top1_acc = t.sum(top_1_pred == labels, dtype=t.float32)
        batch_top2_acc = t.sum(top_2_pred == tiled_2_labels, dtype=t.float32)
        batch_top3_acc = t.sum(top_3_pred == tiled_3_labels, dtype=t.float32)
        batch_top5_acc = t.sum(top_5_pred == tiled_5_labels, dtype=t.float32)                    

        ave_top1_acc += batch_top1_acc.item()
        ave_top2_acc += batch_top2_acc.item()
        ave_top3_acc += batch_top3_acc.item()
        ave_top5_acc += batch_top5_acc.item()                    
    print("total test examples are", total_count)
    running_top1_acc.append(ave_top1_acc / total_count)  # (batch_size * (count_2 + 1)) )
    running_top2_acc.append(ave_top2_acc / total_count)
    running_top3_acc.append(ave_top3_acc / total_count)  # (batch_size * (count_2 + 1)))
    running_top5_acc.append(ave_top5_acc / total_count)  # (batch_size * (count_2 + 1)))                
    # print('Training_size {}--No. of skipped batchess {}'.format(n,skipped_batches))
    print('Average Top-1 accuracy {}'.format( running_top1_acc[-1]))
    print('Average Top-2 accuracy {}'.format( running_top2_acc[-1]))
    print('Average Top-3 accuracy {}'.format( running_top3_acc[-1]))
    print('Average Top-5 accuracy {}'.format( running_top5_acc[-1])) 
    
    print("Saving the predicted value in a csv file")
    file_to_save = f'{save_directory}//best_epoch_eval.csv'
    indx = np.arange(1, len(top1_pred_out)+1, 1)
    df2 = pd.DataFrame()
    df2['index'] = indx                
    df2['link_status'] = gt_beam
    df2['top1_pred'] = top1_pred_out
    df2['top2_pred'] = top2_pred_out
    df2['top3_pred'] = top3_pred_out
    df2['top5_pred'] = top5_pred_out
    df2.to_csv(file_to_save, index=False) 

    
if __name__ == "__main__":
    main()
