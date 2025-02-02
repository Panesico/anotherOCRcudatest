import os
import sys
import time
import random
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from test import validation

# Force CPU device globally
device = torch.device("cpu")

def count_parameters(model):
    print("Modules, Parameters")
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        print(name, param)
        total_params += param
    print(f"Total Trainable Params: {total_params}")
    return total_params

def train(opt, show_number=2, amp=False):
    global device
    device = torch.device("cpu")
    
    # Dataset preparation
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')
    
    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    train_dataset = Batch_Balanced_Dataset(opt)
    
    log = open(f'./saved_models/{opt.experiment_name}/log_dataset.txt', 'a', encoding="utf8")
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, contrast_adjust=opt.contrast_adjust)
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=min(32, opt.batch_size),
        shuffle=True,
        num_workers=int(opt.workers),
        prefetch_factor=512,
        collate_fn=AlignCollate_valid,
        pin_memory=False  # Disabled for CPU usage
    )
    log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()
    
    # Model configuration
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    
    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel,
          opt.output_channel, opt.hidden_size, opt.num_class, opt.batch_max_length,
          opt.Transformation, opt.FeatureExtraction, opt.SequenceModeling, opt.Prediction)
    
    if opt.saved_model != '':
        pretrained_dict = torch.load(opt.saved_model, map_location=device)
        if opt.new_prediction:
            model.Prediction = nn.Linear(model.SequenceModeling_output, len(pretrained_dict['module.Prediction.weight']))
            model = model.to(device)
            print(f'loading pretrained model from {opt.saved_model}')
            if opt.FT:
                model.load_state_dict(pretrained_dict, strict=False)
            else:
                model.load_state_dict(pretrained_dict)
            model.module.Prediction = nn.Linear(model.module.SequenceModeling_output, opt.num_class)
            for name, param in model.module.Prediction.named_parameters():
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)
            model = model.to(device)
        else:
            for name, param in model.named_parameters():
                if 'localization_fc2' in name:
                    print(f'Skip {name} as it is already initialized')
                    continue
                try:
                    if 'bias' in name:
                        init.constant_(param, 0.0)
                    elif 'weight' in name:
                        init.kaiming_normal_(param)
                except Exception as e:
                    if 'weight' in name:
                        param.data.fill_(1)
                        continue
            model = model.to(device)
    else:
        # Weight initialization for a new model
        for name, param in model.named_parameters():
            if 'localization_fc2' in name:
                print(f'Skip {name} as it is already initialized')
                continue
            try:
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)
            except Exception as e:
                if 'weight' in name:
                    param.data.fill_(1)
                    continue
        model = model.to(device)
    
    model.train()
    print("Model:")
    print(model)
    count_parameters(model)
    
    # Setup loss
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
    loss_avg = Averager()
    
    # Freeze some layers if requested
    try:
        if opt.freeze_FeatureFxtraction:
            for param in model.module.FeatureExtraction.parameters():
                param.requires_grad = False
        if opt.freeze_SequenceModeling:
            for param in model.module.SequenceModeling.parameters():
                param.requires_grad = False
    except:
        pass
    
    # Filter parameters that require gradients
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    
    # Setup optimizer
    if opt.optim == 'adam':
        optimizer = optim.Adam(filtered_parameters)
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    print("Optimizer:")
    print(optimizer)
    
    # Save options to log
    with open(f'./saved_models/{opt.experiment_name}/opt.txt', 'a', encoding="utf8") as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)
    
    # Start training
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    i = start_iter
    
    # Although GradScaler and autocast are CUDA-focused, we instantiate GradScaler with amp disabled on CPU.
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    t1 = time.time()
    while True:
        optimizer.zero_grad(set_to_none=True)
        print('training time: ', time.time()-t1)
        t1 = time.time()
        elapsed_time = time.time() - start_time
        
        # Validation
        with open(f'./saved_models/{opt.experiment_name}/log_train.txt', 'a', encoding="utf8") as log:
            model.eval()
            with torch.no_grad():
                valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, \
                infer_time, length_of_data = validation(model, criterion, valid_loader, converter, opt, device)
            model.train()
        
            loss_log = f'[{i}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
            loss_avg.reset()
            current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.4f}'
        
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                torch.save(model.state_dict(), f'./saved_models/{opt.experiment_name}/best_accuracy.pth')
            if current_norm_ED > best_norm_ED:
                best_norm_ED = current_norm_ED
                torch.save(model.state_dict(), f'./saved_models/{opt.experiment_name}/best_norm_ED.pth')
            best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.4f}'
        
            loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
            print(loss_model_log)
            log.write(loss_model_log + '\n')
        
            # Show some predicted results
            dashed_line = '-' * 80
            head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
            predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
            start = random.randint(0, len(labels) - show_number)
            for gt, pred, confidence in zip(labels[start:start+show_number],
                                            preds[start:start+show_number],
                                            confidence_score[start:start+show_number]):
                if 'Attn' in opt.Prediction:
                    gt = gt[:gt.find('[s]')]
                    pred = pred[:pred.find('[s]')]
                predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}'
            print(predicted_result_log)
            log.write(predicted_result_log + '\n')
            print('validation time: ', time.time()-t1)
        
        sys.exit()  # End training loop after one iteration for debugging/testing purposes
        i += 1
