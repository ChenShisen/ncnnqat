# -*- coding:utf-8 -*-
import os
import copy
from ncnnqat import unquant_weight, merge_freeze_bn, register_quantization_hook,save_table
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
from torchvision import datasets,utils
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary






    
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'   
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'   
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'   
    



def net_builder(class_num,net_name="mobile_netv2"):
    if net_name == "mobile_netv2":
        net = models.mobilenet_v2(pretrained=True)
        net.classifier = nn.Sequential(nn.Linear(1280, 1000), nn.ReLU(True),nn.Dropout(0.5),nn.Linear(1000, class_num))
    elif net_name == "resnet18":
        net = models.resnet18(pretrained=True)
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, class_num)
    else:
        raise ValueError("net_name not in(mobile_netv2,resnet18)")
    return net

        
class Mbnet(unittest.TestCase):
    def test(self):
        num_workers = 10

        
        net_name="resnet18"
        net_name="mobile_netv2"
        
        class_num = 10
        
        img_size = 224
        batch_size = 128
        epoch_all = 50
        epoch_merge_bn = epoch_all-5
        
        #maybe cuda out of memery,set test epoch in a small count
        epoch_all = 4
        epoch_merge_bn = epoch_all-2
        
        checkpoint = "./model.pt"
        pre_list = ["train","val"]
        dataloaders = {}
        
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data',
                                                train=True,
                                                download=True,
                                                transform=transform)
        dataloaders['train'] = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=2)
        testset = torchvision.datasets.CIFAR10(root='./data',
                                               train=False,
                                               download=True,
                                               transform=transform)
        dataloaders['val'] = torch.utils.data.DataLoader(testset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=2)
                                                 
                                                 
        
        dummy_input = torch.randn(1, 3, img_size, img_size, device='cuda')
        input_names = [ "input" ]
        output_names = [ "fc" ] #mobilenet
        


        net = net_builder(10,net_name=net_name)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(net)
        net.cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

        best_model_wts = copy.deepcopy(net.state_dict())
        best_acc = 0.0
        best_acc_org = 0.0
        print("training:")
        state_dict_merge = False
        for epoch in range(epoch_all):
            net.train()
            if epoch == epoch_merge_bn:
                best_acc_org = best_acc
                #save not use qat model
                if torch.cuda.device_count() > 1:
                    net_t = net_builder(class_num,net_name=net_name)
                    net_t.cuda()
                    net_t.load_state_dict({k.replace('module.',''):v for k,v in best_model_wts.items()})
                    torch.onnx.export(net_t, dummy_input, "mobilenet_org.onnx", verbose=False, input_names=input_names, output_names=output_names) 
                    print("export org onnx")
                else:
                    torch.onnx.export(net, dummy_input, "mobilenet_org.onnx", verbose=False, input_names=input_names, output_names=output_names) 
                    print("export org onnx")
                register_quantization_hook(net)                
                net = merge_freeze_bn(net)
                
                best_acc = 0.
            if epoch == epoch_merge_bn+1:  
                net = merge_freeze_bn(net)
                print("merge bn")
                best_model_wts = copy.deepcopy(net.state_dict()) #first epoch of qat ,save model as baseline 
            if epoch > epoch_merge_bn+1: 
                print("merge bn")            
                net = merge_freeze_bn(net)
                    
            running_loss = 0.0
            bath_term = 20
            for index, data in enumerate(dataloaders['train']):
                inputs, labels = data
                inputs, labels = Variable(inputs.cuda()), Variable(
                    labels.cuda())
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                if epoch >= epoch_merge_bn:
                    net.apply(unquant_weight)
               
                optimizer.step()

                running_loss += loss.item()
                if index % bath_term == 100:
                    print(' epoch %3d, Iter %5d, loss: %.3f' % (epoch + 1, index + 1, running_loss / bath_term))
                    running_loss = 0.0
            exp_lr_scheduler.step()

            net.eval()
            correct = total = 0
            for data in dataloaders['val']:
                images, labels = data
                outputs = net(Variable(images.cuda()))
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels.cuda()).sum()
                total += labels.size(0)
            print('Epoch: {} Accuracy: {}'.format(str(epoch),str(100.0 * correct.cpu().numpy() / total)))  
            epoch_acc = 100.0 * correct / total
            if epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(net.state_dict()) 
                print("get best ....")          
        net.load_state_dict(best_model_wts) 
        print('Finished Training.')

        net.eval()
        correct = total = 0
        for data in dataloaders['val']:
            images, labels = data
            outputs = net(Variable(images.cuda()))
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels.cuda()).sum()
            total += labels.size(0)
        print('Accuracy: {}'.format(str(100.0 * correct.cpu().numpy() / total)))  

        if torch.cuda.device_count() > 1:
            net_t = net_builder(class_num,net_name=net_name)
            net_t.cuda()
            register_quantization_hook(net_t)
            net_t = merge_freeze_bn(net_t)
            
            net_t.load_state_dict({k.replace('module.',''):v for k,v in net.state_dict().items()})
            torch.onnx.export(net_t, dummy_input, "mobilenet.onnx", verbose=False, input_names=input_names, output_names=output_names) #保存模型
            save_table(net_t,onnx_path="mobilenet.onnx",table="mobilenet.table")
            print("export qat onnx")
        else:
            torch.onnx.export(net, dummy_input, "mobilenet.onnx", verbose=False, input_names=input_names, output_names=output_names)
            save_table(net,onnx_path="mobilenet.onnx",table="mobilenet.table")
            print("export qat onnx")
        print(best_acc_org,best_acc)
if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(Mbnet("test"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
























