import os
import sys
import time
from argparse import ArgumentParser
import math
import numpy as np
import time
import torch
from torch.optim.lr_scheduler import MultiStepLR
import torch.utils.data.distributed
from torchsummary import summary

from ncnnqat import merge_freeze_bn, register_quantization_hook,save_table

from src.model import model, Loss
from src.utils import dboxes300_coco, Encoder

from src.evaluate import evaluate
from src.train import train_loop, tencent_trick
from src.data import *


#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'



class Logger:
    def __init__(self, batch_size, local_rank, n_gpu, print_freq=20):
        self.batch_size = batch_size
        self.local_rank = local_rank
        self.n_gpu = n_gpu
        self.print_freq = print_freq

        self.processed_samples = 0
        self.epochs_times = []
        self.epochs_speeds = []


    def update_iter(self, epoch, iteration, loss):
        if self.local_rank != 0:
            return

        if iteration % self.print_freq == 0:
            print('Epoch: {:2d}, Iteration: {}, Loss: {}'.format(epoch, iteration, loss))

        self.processed_samples = self.processed_samples + self.batch_size

    def start_epoch(self):
        self.epoch_start = time.time()

    def end_epoch(self):
        epoch_time = time.time() - self.epoch_start
        epoch_speed = self.processed_samples / epoch_time

        self.epochs_times.append(epoch_time)
        self.epochs_speeds.append(epoch_speed)
        self.processed_samples = 0

        if self.local_rank == 0:
            print('Epoch {:2d} finished. Time: {:4f} s, Speed: {:4f} img/sec, Average speed: {:4f}'
                .format(len(self.epochs_times)-1, epoch_time, epoch_speed * self.n_gpu, self.average_speed() * self.n_gpu))

    def average_speed(self):
        return sum(self.epochs_speeds) / len(self.epochs_speeds)


def make_parser():
    epoch_all = 65
    epoch_qat = epoch_all-5 if epoch_all-5>0 else epoch_all
    
    eval_list = [0,epoch_all-1] if epoch_all-1>0 else [0]
    
    parser = ArgumentParser(
        description="Train Single Shot MultiBox Detector on COCO")
    parser.add_argument(
        '--data', '-d', type=str, default='./data/coco', required=False,
        help='path to test and training data files')
    parser.add_argument(
        '--epochs', '-e', type=int, default=epoch_all,  #65
        help='number of epochs for training')
    parser.add_argument(
        '--qat-epoch', '-q', type=int, default=epoch_qat,
        help='epoch of qat begaining')
    parser.add_argument(
        '--batch-size', '--bs', type=int, default=32,
        help='number of examples for each iteration')
    parser.add_argument(
        '--eval-batch-size', '--ebs', type=int, default=32,
        help='number of examples for each evaluation iteration')
    parser.add_argument(
        '--seed', '-s', type=int, default=0,
        help='manually set random seed for torch')
    parser.add_argument(
        '--evaluation', nargs='*', type=int,
        default=eval_list,#[0, 48, 53, 59,63, 64,65],
        help='epochs at which to evaluate')
    parser.add_argument(
        '--multistep', nargs='*', type=int, default=[43, 54],
        help='epochs at which to decay learning rate')
    parser.add_argument(
        '--target', type=float, default=None,
        help='target mAP to assert against at the end')
        
    #save model
    parser.add_argument('--check-save', '--s', type=bool, default=True)
    parser.add_argument(
        '--check-point', '-c', type=str, default='./models', required=False,
        help='path to model save files')
    parser.add_argument('--onnx_save', action='store_true')

    # Hyperparameters
    parser.add_argument(
        '--learning-rate', '--lr', type=float, default=2.6e-3, help='learning rate')
    parser.add_argument(
        '--momentum', '-m', type=float, default=0.9,
        help='momentum argument for SGD optimizer')
    parser.add_argument(
        '--weight-decay', '--wd', type=float, default=0.0005,
        help='momentum argument for SGD optimizer')
    parser.add_argument('--warmup', type=int, default=None)
    parser.add_argument(
        '--backbone', type=str, default='resnet18',
        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--fp16-mode', type=str, default='static', choices=['off', 'static', 'amp'],
        help='Half precission mode to use')

    # Distributed
    parser.add_argument('--local_rank', default=0, type=int,
        help='Used for multi-process training. Can either be manually set ' +
            'or automatically set by using \'python -m multiproc\'.')

    # Pipeline control
    parser.add_argument(
        '--data_pipeline', type=str, default='dali', choices=['dali', 'no_dali'],
        help='data preprocessing pipline to use')

    return parser


def train(args):

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        print('WORLD_SIZE in os.environ',os.environ['WORLD_SIZE'],args.local_rank)
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        print(args.distributed)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.N_gpu = torch.distributed.get_world_size()
    else:
        args.N_gpu = 1

    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)
    cocoGt = get_coco_ground_truth(args)
    
    ssd300 = model(args)
    
    loss_func = Loss(dboxes)
    loss_func.cuda()
    
    
    args.learning_rate = args.learning_rate * args.N_gpu * (args.batch_size / 32)
    iteration = 0
    
    optimizer = torch.optim.SGD(
        tencent_trick(ssd300), 
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    scheduler = MultiStepLR(
        optimizer=optimizer, 
        milestones=args.multistep, 
        gamma=0.1)

    

    val_dataloader, inv_map = get_val_dataloader(args)
    train_loader = get_train_loader(args, dboxes)
    
    #print(inv_map)
    #print(val_dataset.label_info)
    
    acc = 0
    acc_best = 0
    epoch_check = 0
    logger = Logger(args.batch_size, args.local_rank, args.N_gpu)
    
    for epoch in range(epoch_check, args.epochs):
        logger.start_epoch()
        #scheduler.step()
        #print(ssd300)
        '''qat'''
        if epoch==args.qat_epoch:
            register_quantization_hook(ssd300)                
            ssd300 = merge_freeze_bn(ssd300)
            print("qat hook...")
        if epoch>args.qat_epoch:
            ssd300 = merge_freeze_bn(ssd300)
            print("merge bn ...")
        '''qat'''  
            
        iteration = train_loop(
            ssd300, loss_func, epoch, optimizer, 
            train_loader, iteration, logger, args)
        scheduler.step()
        logger.end_epoch()

        if epoch in args.evaluation:
            acc = evaluate(ssd300, val_dataloader, cocoGt, encoder, inv_map, args)
            if args.local_rank == 0:
                print('Epoch {:2d}, Accuracy: {:4f} mAP'.format(epoch, acc))
                
            
            if acc>=acc_best and args.local_rank == 0:
                acc_best = acc
                
                if args.distributed:
                    model_dict = ssd300.module.state_dict()
                else:
                    model_dict = ssd300.state_dict()
                torch.save({
                'epoch': epoch+1,
                'model_state_dict': model_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc':acc_best,
                "inv_map":inv_map,
                "scheduler":scheduler.state_dict(),
                }, args.checkpoint)

        if args.data_pipeline == 'dali':
            train_loader.reset()

    return acc, logger.average_speed()
        

if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    
    
    if args.onnx_save:  #after train  ,load model , save onnx model and ncnn table
        #python main.py --onnx_save
        onnx_path = os.path.join(args.check_point,"model.onnx")
        table_path = os.path.join(args.check_point,"model.table")
        #print(onnx_path)
        checkpoint = os.path.join(args.check_point,"model.pt")
        #print(checkpoint)
        ssd300 = model(args,onnx_save=args.onnx_save)
        summary(ssd300, input_size=(3, 300, 300), device='cpu')
        ssd300.cuda()
        '''qat'''
        register_quantization_hook(ssd300)
        ssd300 = merge_freeze_bn(ssd300)
        '''qat'''
        if os.path.exists(checkpoint):
            print("loadmodel from checkpoint...")
            checkpoint_load = torch.load(checkpoint,map_location='cpu')
            #ssd300.module.load_state_dict(checkpoint_load['model_state_dict']) #donot know ssd300 is distributed
            ssd300.load_state_dict({k.replace('module.',''):v for k,v in checkpoint_load['model_state_dict'].items()})   
            print("loadmodel from checkpoint end...")
        ssd300.eval()
        input_names = [ "input" ]
        #output_names = [ "SSD300-184" ]
        output_names = [ "Conv2d-93" ]
        dummy_input = torch.ones([1, 3, 300, 300]).cuda()
        #dummy_input = torch.randn(1, 3, 300, 300, device='cuda')
        torch.onnx.export(ssd300, dummy_input, onnx_path, verbose=False, input_names=input_names, output_names=output_names)
        save_table(ssd300,onnx_path=onnx_path,table=table_path)
    else:
        args.checkpoint = os.path.join(args.check_point,"model.pt")
        if args.local_rank == 0:
            os.makedirs(args.check_point, exist_ok=True)

        torch.backends.cudnn.benchmark = True

        if args.fp16_mode != 'off':
            args.fp16 = True
        else:
            args.fp16 = False
        #print(args)
        start_time = time.time()
        acc, avg_speed = train(args)
        # avg_speed is reported per node, adjust for the global speed
        try:
            num_shards = torch.distributed.get_world_size()
        except RuntimeError:
            num_shards = 1
        avg_speed = num_shards * avg_speed
        training_time = time.time() - start_time

        if args.local_rank == 0:
            print("Training end: Average speed: {:3f} img/sec, Total time: {:3f} sec, Final accuracy: {:3f} mAP"
              .format(avg_speed, training_time, acc))

            if args.target is not None:
                if args.target > acc:
                    print('Target mAP of {} not met. Possible regression'.format(args.target))
                    sys.exit(1)
'''
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.253
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.429
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.262
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.075
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.273
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.397
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.240
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.349
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.367
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.122
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.406
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.551
 214img/sec
 
 dali 218img/sec
 
 
 warmup 200 + dali + fp16
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.256
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.434
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.263
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.078
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.274
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.408
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.240
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.352
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.368
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.126
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.403
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.563
Current AP: 0.25628
Epoch 64, Accuracy: 0.256285 mAP
DONE (t=9.45s).
Training end: Average speed: 232.580538 img/sec, Total time: 35018.003625 sec, Final accuracy: 0.256285 mAP



not qat
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.193
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.344
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.191
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.042
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.195
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.328
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.199
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.293
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.309
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.084
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.326
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.501
Current AP: 0.19269

qat  resnet18
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.192
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.342
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.194
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.041
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.194
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.327
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.197
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.291
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.307
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.082
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.325
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.497
Current AP: 0.19202
'''