import os.path
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data
import json
from loss import DiceLoss

from tqdm import tqdm

def config_loader(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def create_data_loaders(image_datasets, train_domains, test_domains, batch_size=32, shuffle=True, train_portion=0.8,
                        num_workers=0):
    source_dataset = torch.utils.data.ConcatDataset([image_datasets[x] for x in train_domains])

    classes = image_datasets[train_domains[0]].classes

    source_size = len(source_dataset)
    train_size = int(source_size * train_portion)
    valid_size = source_size - train_size

    train_dataset, valid_dataset = torch.utils.data.random_split(source_dataset, [train_size, valid_size])
    test_dataset = image_datasets[test_domains]

    # do not use tansforms for the test data
    test_dataset.apply_transform = False

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                                               num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle,
                                               num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=num_workers)



    return train_loader, valid_loader, test_loader, classes


def train_model(number_of_epochs, model, train_loader, valid_loader, test_loader, model_directory, lr=0.001,
                reconstruction=True, device='cpu', lmda=1, model_name=""):
    
    model = model.to(device)

    ae_criterion = DiceLoss().to(device)
    cl_criterion = nn.CrossEntropyLoss().to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr,
        momentum=0.9,
        weight_decay=0.0005)

    summary = {
        "train": {
            "total": [],
            "classification": [],
            "reconstruction": []
        },
        "val": {
            "total": [],
            "classification": [],
            "reconstruction": [],
            "accuracy": []
        },
        "test": {
            "total": [],
            "classification": [],
            "reconstruction": [],
            "accuracy": []
        }
    }

    for epoch in range(number_of_epochs):

        model.train()
        train_losses = [0., 0., 0.]  # [overall loss, first loss, second loss if reconstruction==True]
        t1 = time.time()

        for i, batch in enumerate(train_loader):
            # print(f'iteration {i} in epoch {epoch}')
            optimizer.zero_grad()
            if reconstruction:
                samples, canny_samples, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)

                predicted_labels, reconstructed_canny = model(samples)

                loss_ae = ae_criterion(reconstructed_canny, canny_samples)
                # print (predicted_labels)
                loss_cl = cl_criterion(predicted_labels, labels)
                loss = loss_cl + lmda * loss_ae

                loss.backward()
                optimizer.step()

                train_losses[0] += loss.item()
                train_losses[1] += loss_cl.item()
                train_losses[2] += loss_ae.item()

            else:

                samples, labels = batch[0].to(device), batch[1].to(device)

                predicted_labels = model(samples)
                loss = cl_criterion(predicted_labels, labels)

                loss.backward()
                optimizer.step()

                train_losses[0] += loss.item()
                train_losses[1] += loss.item()

        # Evaluate on validation dataset
        model.eval()

        val_losses = [0., 0., 0.]  # [overall loss, first loss, second loss if reconstruction==True]
        all_pred = []
        all_lb = []

        for i, batch in enumerate(valid_loader):

            with torch.no_grad():

                if reconstruction:
                    samples, canny_samples, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                    predicted_labels, reconstructed_canny = model(samples)
                    val_loss_ae = ae_criterion(reconstructed_canny, canny_samples)
                    val_loss_cl = cl_criterion(predicted_labels, labels)
                    val_loss = val_loss_cl + lmda * val_loss_ae
                    val_losses[0] += val_loss.item()
                    val_losses[1] += val_loss_cl.item()
                    val_losses[2] += val_loss_ae.item()

                else:
                    samples, labels = batch[0].to(device), batch[1].to(device)
                    predicted_labels = model(samples)
                    val_loss = cl_criterion(predicted_labels, labels)
                    val_losses[0] += val_loss.item()
                    val_losses[1] += val_loss.item()

                logits = torch.nn.functional.softmax(predicted_labels, dim=-1)
                all_pred.extend(torch.argmax(logits, dim=-1))
                all_lb.extend(labels)

        val_accuracy = torch.mean((torch.tensor(all_pred) == torch.tensor(all_lb)).float()).item()

        # Evaluate on test dataset

        test_losses = [0., 0., 0.]  # [overall loss, first loss, second loss if reconstruction==True]
        all_pred = []
        all_lb = []

        for i, batch in enumerate(test_loader) :

            with torch.no_grad():

                if reconstruction:
                    samples, canny_samples, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                    predicted_labels, reconstructed_canny = model(samples)
                    test_loss_ae = ae_criterion(reconstructed_canny, canny_samples)
                    test_loss_cl = cl_criterion(predicted_labels, labels)
                    test_loss = test_loss_cl + lmda * test_loss_ae
                    test_losses[0] += test_loss.item()
                    test_losses[1] += test_loss_cl.item()
                    test_losses[2] += test_loss_ae.item()

                else:
                    samples, labels = batch[0].to(device), batch[1].to(device)
                    predicted_labels = model(samples)
                    test_loss = cl_criterion(predicted_labels, labels)
                    test_losses[0] += test_loss.item()
                    test_losses[1] += test_loss.item()

                logits = torch.nn.functional.softmax(predicted_labels, dim=-1)
                all_pred.extend(torch.argmax(logits, dim=-1))
                all_lb.extend(labels)

        test_accuracy = torch.mean((torch.tensor(all_pred) == torch.tensor(all_lb)).float()).item()

        #  4. Log results and save model checkpoints

        summary["train"]["total"].append(train_losses[0] / len(train_loader))
        summary["train"]["classification"].append(train_losses[1] / len(train_loader))
        summary["train"]["reconstruction"].append(train_losses[2] / len(train_loader))

        summary["val"]["total"].append(val_losses[0] / len(valid_loader))
        summary["val"]["classification"].append(val_losses[1] / len(valid_loader))
        summary["val"]["reconstruction"].append(val_losses[2] / len(valid_loader))
        summary["val"]["accuracy"].append(val_accuracy * 100)

        summary["test"]["total"].append(test_losses[0] / len(test_loader))
        summary["test"]["classification"].append(test_losses[1] / len(test_loader))
        summary["test"]["reconstruction"].append(test_losses[2] / len(test_loader))
        summary["test"]["accuracy"].append(test_accuracy * 100)

        print("{} ----------- Epoch: {}/{} ~time: {:.3f}".format(model_name, epoch + 1, number_of_epochs, time.time() - t1))
        print("Train Loss: {:.5f}  1st Loss:   {:.5f}  2nd Loss {:.5f}".format(train_losses[0] / len(
            train_loader), train_losses[1] / len(train_loader), train_losses[2] / len(train_loader)))
        print("Validation Loss: {:.5f}  1st Loss:   {:.5f}  2nd Loss {:.5f}".format(val_losses[0] / len(
            valid_loader), val_losses[1] / len(valid_loader), val_losses[2] / len(valid_loader)))
        print("Test Loss: {:.5f}  1st Loss:   {:.5f}  2nd Loss {:.5f}".format(test_losses[0] / len(
            test_loader), test_losses[1] / len(test_loader), test_losses[2] / len(test_loader)))

        print("Val Acc {:.5f}".format(val_accuracy))
        print("Test Acc {:.5f}".format(test_accuracy))

        this_model_path = os.path.join(model_directory, 'model_{}.pt'.format(epoch + 1))
        if epoch % 5 == 0:
            torch.save(model.state_dict(), this_model_path)

        summary_path = os.path.join(model_directory, "summary.json")

        with open(summary_path, "w") as f:
            json.dump(summary, f)
            f.close()

    return summary


def draw_summary(summary_path):
    with open(summary_path) as json_file:
        summary = json.load(json_file)
        json_file.close()

    # print(summary)
    train_loss = summary['train']['total']
    val_loss = summary['val']['total']
    val_accuracy = summary['val']['accuracy']

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].set_title("losses")
    ax[0].set(xlabel='n_epoch', ylabel='loss')
    ax[0].plot(train_loss, label="train loss")

    ax[0].plot(val_loss, label="validation loss")
    ax[0].legend()

    ax[1].set_title("validation accuracy")
    ax[1].set(xlabel='n_epoch', ylabel='accuracy')
    ax[1].plot(val_accuracy)

    plt.subplots_adjust(wspace=1, hspace=1)
    plt.show()
