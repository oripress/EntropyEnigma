import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.models as models
import torchvision.transforms as trn

from rdumb import RDumb
from temperature_scaling import ModelWithTemperature


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # For ImageNet variants
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    test_set = dset.ImageFolder(args.data, transform=trn.Compose(
        [trn.ToTensor(), trn.Normalize(mean, std)]))

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.bs, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True).to(device)
    model = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    model.eval()

    target_accuracy = get_acc(model, test_loader)

    if args.valdata != None:
        val_set = dset.ImageFolder(args.valdata, transform=trn.Compose(
            [trn.Resize(256), trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)]))

        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=args.bs, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True)

        model = calibrate(model, val_loader, args.calibrated_model)

    model = RDumb(model)
    stored_images, stored_labels = get_unlabeled_subset(model.model, test_loader)

    iterations = 0
    df = None

    while iterations < 990:
        for i, (batch) in enumerate(test_loader):
            images, labels = batch

            if iterations == 0 or iterations == 990:
                df = check_flips(model.model, stored_images, stored_labels, iterations, device, df)

            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            _ = model(images)
            iterations += 1

    # Coefficients for the flips to accuracy function $f$
    # Weighted deg 2 - Best Results
    coefficients = [3.55553868e-04, -3.21445023e-01, 7.56618580e+01]
    # Unweighted deg 1 - Second Best
    coefficients_2 = [-0.10318227, 83.55158891]
    # Unweighted deg 2 - Third Best
    coefficients_3 = [-1.01165493e-05, -9.59406999e-02, 8.25756446e+01]

    # Create polynomial functions from coefficients
    f = np.poly1d(coefficients)
    f2 = np.poly1d(coefficients_2)
    f3 = np.poly1d(coefficients_3)

    # Assume count_flips function definition and target_accuracy variable provided elsewhere
    weighted_flips = count_flips(df)
    unweighted_flips = count_flips(df, weighted=False)

    # Calculate estimated accuracies
    estimated_accuracy = f(weighted_flips)
    estimated_accuracy_2 = f2(unweighted_flips)
    estimated_accuracy_3 = f3(unweighted_flips)

    print(f"Target Accuracy: {target_accuracy:.2f}")
    print(f"Estimated Accuracy using a quadratic f with weighted flips: {estimated_accuracy:.2f}")
    print(f"Estimated Accuracy using a linear f with unweighted flips: {estimated_accuracy_2:.2f}")
    print(f"Estimated Accuracy using a quadratic f with unweighted flips: {estimated_accuracy_3:.2f}")


def check_flips(model, stored_images, stored_labels, iteration, device, df=None):
    total_seen_so_far = 0
    batch_size = 100
    model.eval()

    iteration_list = []
    image_index_list = []
    top_class_list = []
    confidence_list = []

    with torch.no_grad():
        for i in range(0, stored_images.size(0), batch_size):
            images, labels = stored_images[i:i + batch_size], stored_labels[i:i + batch_size]
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            output = model(images)

            num_images_in_batch = images.size(0)
            total_seen_so_far += num_images_in_batch

            top_val, top_idx = torch.max(F.softmax(output, dim=1), dim=1)
            for j in range(num_images_in_batch):
                iteration_list.append(iteration)
                image_index_list.append(total_seen_so_far - num_images_in_batch + j)
                top_class_list.append(top_idx[j].item())
                confidence_list.append(top_val[j].item())

    # Build or append to the DataFrame
    new_predictions_df = pd.DataFrame({
        'Iteration': iteration_list,
        'Image_Index': image_index_list,
        'Top_Class': top_class_list,
        'Confidence': confidence_list,
    })

    model.train()

    if df is not None:
        return pd.concat([df, new_predictions_df], ignore_index=True)
    else:
        return new_predictions_df


def get_acc(model, loader):
    num_correct = 0
    total_seen_so_far = 0
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            output = model(images)
            vals, pred = (output).max(dim=1, keepdim=True)
            correct_this_batch = pred.eq(labels.view_as(pred)).sum().item()
            num_correct += correct_this_batch

            num_images_in_batch = images.size(0)
            total_seen_so_far += num_images_in_batch

    model.train()
    return 100 * float(num_correct) / total_seen_so_far


def get_unlabeled_subset(model, loader):
    total_seen_so_far = 0
    confidence = 0
    image_list = []
    stored_images, stored_labels = [], []
    total_images = 1000
    bs = 100

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            images, _ = batch
            image_list.append(images)
            if len(image_list) >= int(total_images / images.size(0)) + 1:
                break
        images = torch.cat(image_list)[:total_images]

        for i in range(int(total_images / bs)):
            cur_images = images[i * bs: (i + 1) * bs].cuda(non_blocking=True)
            output = model(cur_images)
            num_images_in_batch = images.size(0)
            total_seen_so_far += num_images_in_batch

            vals, pred = (output).max(dim=1)
            confidence += vals.sum().item()

            stored_images.append(cur_images.cpu())
            stored_labels.append(pred.cpu().view(-1))

    stored_images = torch.cat(stored_images).cpu()
    stored_labels = torch.cat(stored_labels).cpu()

    model.train()

    return stored_images, stored_labels


def calibrate(model, val_loader, model_path):
    if os.path.exists(model_path):
        with open(model_path, 'r') as f:
            temp = json.load(f)

        model = ModelWithTemperature(model)
        model.set_temperature(temp)
    else:
        model = ModelWithTemperature(model)
        model.find_temperature(val_loader)
        temp = model.temperature.item()

        with open(model_path, 'w') as f:
            json.dump(temp, f)
    return model


def count_flips(df, end=990, weighted=True):
    # Filter out rows for iteration 0 and the specified end iteration
    df_iteration_end = df[df['Iteration'] == end]
    if df_iteration_end.empty:
        return None

    start = 0
    df = df[df['Iteration'].isin([start, end])]

    # Get the top class for each image based on the highest confidence
    idx = df.groupby(['Image_Index', 'Iteration'])['Confidence'].idxmax()
    df = df.loc[idx]

    # Pivot the dataframe to get labels at iteration 0 and end iteration side-by-side
    df_pivot = df.pivot(index='Image_Index', columns='Iteration', values='Top_Class')

    # Indicator for whether label flipped or not
    df_pivot['Label_Flipped'] = (df_pivot[start] != df_pivot[end]).astype(int)

    if weighted:
        initial_confidence = df[df['Iteration'] == start].set_index('Image_Index')['Confidence']
        # Compute the percentile of each initial confidence
        confidence_percentile = initial_confidence.rank() / len(initial_confidence)
        weight = confidence_percentile
    else:
        weight = 1

    # Multiply by confidence percentile or uniform weight and then sum
    weighted_sum = (df_pivot['Label_Flipped'] * weight).sum()
    return weighted_sum


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed for initialization.')
    parser.add_argument('--bs', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--data', type=str, help='Path to the dataset directory.')
    parser.add_argument('--valdata', default=None, type=str,
                        help='Path to validation images, to be used in calibration.')
    parser.add_argument('--calibrated_model', default='./model.json', type=str,
                        help='Path to the calibrated model JSON (used in caching calibrated models).')

    args = parser.parse_args()
    main(args)
