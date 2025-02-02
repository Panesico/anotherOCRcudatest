import os
import sys
import time
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
from torch.cuda.amp import autocast, GradScaler
import numpy as np

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from test import validation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_parameters(model):
    print("Modules, Parameters")
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        #table.add_row([name, param])
        total_params+=param
        print(name, param)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def train(opt, show_number=2, amp=False):
    global device
    device = torch.device("cpu")  # Force CPU usage

    """ Dataset Preparation """
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
        valid_dataset, batch_size=min(32, opt.batch_size),
        shuffle=True, num_workers=int(opt.workers),
        prefetch_factor=512, collate_fn=AlignCollate_valid, pin_memory=False  # Set pin_memory=False for CPU
    )
    log.write(valid_dataset_log)
    log.write('-' * 80 + '\n')
    log.close()

    """ Model Configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    opt.input_channel = 3 if opt.rgb else 1
    model = Model(opt).to(device)  # Ensure model is on CPU

    print('Model input parameters:', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel,
          opt.output_channel, opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation,
          opt.FeatureExtraction, opt.SequenceModeling, opt.Prediction)

    if opt.saved_model != '':
        pretrained_dict = torch.load(opt.saved_model, map_location=device)  # Load model on CPU
        if opt.new_prediction:
            model.Prediction = nn.Linear(model.SequenceModeling_output, len(pretrained_dict['module.Prediction.weight']))
        model.load_state_dict(pretrained_dict, strict=not opt.FT)

        if opt.new_prediction:
            model.Prediction = nn.Linear(model.SequenceModeling_output, opt.num_class)
            for name, param in model.Prediction.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.kaiming_normal_(param)

    else:
        for name, param in model.named_parameters():
            if 'localization_fc2' in name:
                print(f'Skip {name} as it is already initialized')
                continue
            try:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.kaiming_normal_(param)
            except Exception:
                if 'weight' in name:
                    param.data.fill_(1)

    model.train()
    print("Model:")
    print(model)
    count_parameters(model)

    """ Setup Loss """
    criterion = nn.CTCLoss(zero_infinity=True).to(device) if 'CTC' in opt.Prediction else nn.CrossEntropyLoss(ignore_index=0).to(device)

    """ Freeze Some Layers """
    if getattr(opt, 'freeze_FeatureExtraction', False):
        for param in model.FeatureExtraction.parameters():
            param.requires_grad = False
    if getattr(opt, 'freeze_SequenceModeling', False):
        for param in model.SequenceModeling.parameters():
            param.requires_grad = False

    """ Filter Trainable Parameters """
    filtered_parameters = [p for p in model.parameters() if p.requires_grad]
    params_num = [np.prod(p.size()) for p in filtered_parameters]
    print('Trainable params num:', sum(params_num))

    """ Setup Optimizer """
    optimizer = optim.Adam(filtered_parameters) if opt.optim == 'adam' else optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    print("Optimizer:")
    print(optimizer)

    """ Save Options """
    with open(f'./saved_models/{opt.experiment_name}/opt.txt', 'a', encoding="utf8") as opt_file:
        opt_log = '------------ Options -------------\n'
        for k, v in vars(opt).items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)

    """ Start Training """
    start_iter = 0
    if opt.saved_model:
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print(f'Continuing training from iteration: {start_iter}')
        except:
            pass

    start_time = time.time()
    best_accuracy, best_norm_ED, i = -1, -1, start_iter
    scaler = GradScaler()

    while True:
        optimizer.zero_grad(set_to_none=True)
        elapsed_time = time.time() - start_time

        """ Validation """
        model.eval()
        with torch.no_grad():
            valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, infer_time, length_of_data = validation(
                model, criterion, valid_loader, converter, opt, device)
        model.train()

        """ Logging """
        loss_log = f'[{i}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
        loss_avg.reset()
        current_model_log = f'Current_accuracy: {current_accuracy:0.3f}, Current_norm_ED: {current_norm_ED:0.4f}'

        """ Save Best Model """
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            torch.save(model.state_dict(), f'./saved_models/{opt.experiment_name}/best_accuracy.pth')
        if current_norm_ED > best_norm_ED:
            best_norm_ED = current_norm_ED
            torch.save(model.state_dict(), f'./saved_models/{opt.experiment_name}/best_norm_ED.pth')
        best_model_log = f'Best_accuracy: {best_accuracy:0.3f}, Best_norm_ED: {best_norm_ED:0.4f}'

        loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
        print(loss_model_log)

        """ Show Some Predictions """
        dashed_line = '-' * 80
        head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
        predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'

        start = random.randint(0, len(labels) - show_number)
        for gt, pred, confidence in zip(labels[start:start+show_number], preds[start:start+show_number], confidence_score[start:start+show_number]):
            if 'Attn' in opt.Prediction:
                gt, pred = gt.split('[s]')[0], pred.split('[s]')[0]
            predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
        predicted_result_log += f'{dashed_line}'

        print(predicted_result_log)
        sys.exit()
        i += 1
