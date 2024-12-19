import json
from datetime import datetime

from data_loader.forecast_dataloader import ForecastDataset, de_normalized
from models.base_model import Model
import torch
import torch.nn as nn
import torch.utils.data as torch_data
import numpy as np
import time
import os
from datetime import datetime

from utils.math_utils import evaluate


def save_model(model, model_dir, epoch=None):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_stemgnn.pt')
    with open(file_name, 'wb') as f:
        torch.save(model, f)


def load_model(model_dir, epoch=None):
    if not model_dir:
        return
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_stemgnn.pt')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(file_name):
        return
    with open(file_name, 'rb') as f:
        model = torch.load(f)
    return model


def inference(model, dataloader, device, node_cnt, window_size, horizon):
    all_predicted_labels = []
    all_target_labels = []
    
    model.eval()
    with torch.no_grad():
        for i, (inputs, target) in enumerate(dataloader):
            inputs = inputs.to(device)
            target = target.to(device)
            
            # store target labels
            all_target_labels.append(target.view(-1).detach().cpu().numpy())
            
            # perform predictions
            forecast, _ = model(inputs)  # [batch_size, time_steps, num_classes]
            
            # get predicted labels (argmax over classes)
            predicted_labels = torch.argmax(forecast, dim=-1)  # (batch_size, time steps)
            all_predicted_labels.append(predicted_labels.view(-1).detach().cpu().numpy())
    
    # concatenate all predictions and targets for the entire dataset
    all_predicted_labels = np.concatenate(all_predicted_labels, axis=0)  
    all_target_labels = np.concatenate(all_target_labels, axis=0)        
    
    return all_predicted_labels, all_target_labels



def validate(model, dataloader, device, normalize_method, statistic,
             node_cnt, window_size, horizon,
             result_file=None):
    start = datetime.now()

    # get predicted and target labels using inference
    predicted_labels, target_labels = inference(model, dataloader, device,
                                                node_cnt, window_size, horizon)
    
    # save the target and predicted labels to CSV files
    if result_file:
        if not os.path.exists(result_file):
            os.makedirs(result_file)
        
        # save target and predicted labels
        np.savetxt(f'{result_file}/target.csv', target_labels, delimiter=",")
        np.savetxt(f'{result_file}/predict.csv', predicted_labels, delimiter=",")
    
    # calculate evaluation metrics (f1_per_class, accuracy_per_class, f1_overall, accuracy_overall)
    score = evaluate(target_labels, predicted_labels, 3)

    # Return f1_overall and accuracy_overall
    f1_overall = score['f1_overall']
    accuracy_overall = score['accuracy_overall']
    
    # print out scores
    end = datetime.now()
    print(f'Validation took: {end - start}')
    print(f'Overall F1 Score: {f1_overall:.4f}')
    print(f'Overall Accuracy: {accuracy_overall:.4f}')
    
    return {'f1_overall': f1_overall, 'accuracy_overall': accuracy_overall}



def train(train_data, target_train, args, result_file, val_data, val_target):
    node_cnt = train_data.shape[1]
    model = Model(node_cnt, 2, args.window_size, args.multi_layer, horizon=args.horizon)
    model.to(args.device)
    if len(train_data) == 0:
        raise Exception('Cannot organize enough training data')
    if len(val_data) == 0:
        raise Exception('Cannot organize enough validatio data')

    if args.norm_method == 'z_score':
        train_mean = np.mean(train_data, axis=0)
        train_std = np.std(train_data, axis=0)
        normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
    elif args.norm_method == 'min_max':
        train_min = np.min(train_data, axis=0)
        train_max = np.max(train_data, axis=0)
        normalize_statistic = {"min": train_min.tolist(), "max": train_max.tolist()}
    else:
        normalize_statistic = None
    if normalize_statistic is not None:
        with open(os.path.join(result_file, 'norm_stat.json'), 'w') as f:
            json.dump(normalize_statistic, f)

    if args.optimizer == 'RMSProp':
        my_optim = torch.optim.RMSprop(params=model.parameters(), lr=args.lr, eps=1e-08)
    else:
        my_optim = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)

    train_set = ForecastDataset(train_data, target_train, window_size=args.window_size, horizon=args.horizon,
                                normalize_method=args.norm_method, norm_statistic=normalize_statistic)
    train_loader = torch_data.DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=True,
                                         num_workers=0)
    val_set = ForecastDataset(val_data, val_target, window_size=args.window_size, horizon=args.horizon,
                               normalize_method=args.norm_method, norm_statistic=normalize_statistic)
    val_loader = torch_data.DataLoader(val_set, batch_size=args.batch_size, drop_last=False,
                                        shuffle=False, num_workers=0)
    

    forecast_loss = nn.CrossEntropyLoss().to(args.device)  # For multi-class classificatio

    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params += param
    print(f"Total Trainable Params: {total_params}")

    f1_best = 0
    validate_score_non_decrease_count = 0
    performance_metrics = {}
    for epoch in range(args.epoch):
        epoch_start_time = time.time()
        model.train()
        loss_total = 0
        cnt = 0
        for i, (inputs, target) in enumerate(train_loader):
            inputs = inputs.to(args.device)
            target = target.to(args.device)

            # convert target labels to zero-indexed (1->0, 2->1, 3->2)
            target = target - 1

            model.zero_grad()
            forecast, _ = model(inputs)

            #print(f"forecast.shape: {forecast.shape}")
            #print(f"target.shape: {target.shape}")
            loss = forecast_loss(forecast.view(-1, 3), target.view(-1))  # Ensure shapes match
            cnt += 1
            loss.backward()

            my_optim.step()
            loss_total += float(loss)
        print('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f}'.format(epoch+1, (
                time.time() - epoch_start_time), loss_total / cnt))
        save_model(model, result_file, epoch)
        if (epoch+1) % args.exponential_decay_step == 0:
            my_lr_scheduler.step()
        if (epoch + 1) % args.validate_freq == 0:
            is_best_for_now = False
            print('------ validate on data: VALIDATION ------')
            performance_metrics = \
                validate(model, val_loader, args.device, args.norm_method, normalize_statistic,
                      node_cnt, args.window_size, args.horizon,
                      result_file=result_file)
            if f1_best < performance_metrics['f1_overall']:
                f1_best = performance_metrics['f1_overall']
                is_best_for_now = True
                validate_score_non_decrease_count = 0
            else:
                validate_score_non_decrease_count += 1
            # save model
            if is_best_for_now:
                save_model(model, result_file)
        # early stop
        if args.early_stop and validate_score_non_decrease_count >= args.early_stop_step:
            break
    return performance_metrics, normalize_statistic


def test(test_data, target_data, args, result_train_file, result_test_file):
    with open(os.path.join(result_train_file, 'norm_stat.json'),'r') as f:
        normalize_statistic = json.load(f)
    model = load_model(result_train_file)
    node_cnt = test_data.shape[1]
    test_set = ForecastDataset(test_data, target_data, window_size=args.window_size, horizon=args.horizon,
                               normalize_method=args.norm_method, norm_statistic=normalize_statistic)
    test_loader = torch_data.DataLoader(test_set, batch_size=args.batch_size, drop_last=False,
                                        shuffle=False, num_workers=0)
    print(f'Performance on test set:')
    performance_metrics = validate(model, test_loader, args.device, args.norm_method, normalize_statistic,
                      node_cnt, args.window_size, args.horizon,
                      result_file=result_test_file)

    
