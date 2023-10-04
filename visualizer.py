
import os
import matplotlib.pyplot as plt
import json
import random

from custom_dataloder import DGDataset
from models import ResNetAutoEncoderClassifier
import torch, torchvision

from utils import create_data_loaders, config_loader

def draw_summary(summary_path, output_path):
    with open(summary_path) as json_file:
        summary = json.load(json_file)
        json_file.close()

    # print(summary)
    train_loss = summary['train']['total']
    
    val_loss = summary['val']['total']
    test_loss = summary['test']['total']
    
    val_accuracy = summary['val']['accuracy']
    test_accuracy = summary['test']['accuracy']

    print(f'max test acc for {summary_path} is :{max(test_accuracy)} in epoch {test_accuracy.index(max(test_accuracy))}')
    print(f'max val acc for {summary_path} is :{max(val_accuracy)} in epoch {val_accuracy.index(max(val_accuracy))}')
    
    fig, ax = plt.subplots(3, 1, sharex=True)
    # ax[0].set_title("losses")
    # ax[0].set(xlabel='n_epoch', ylabel='loss')
    
    # # ax[0].plot(train_loss, label="train loss")
    # # ax[0].plot(val_loss, label="validation loss")
    # ax[0].plot(test_loss, label="test loss")
    
    # ax[0].legend()
    
    set = 'test'
    set_loss = summary[set]['total']
    set_cls_loss = summary[set]['classification']
    set_rec_loss = summary[set]['reconstruction']
    
    ax[0].set_title(f'{set} loss')
    
    ax[0].plot(set_loss, label = 'total')
    # ax[2].plot(set_cls_loss, label = 'cls')
    # ax[2].plot(set_rec_loss, label = 'rec')
    
    ax[0].legend()
    
    
# 
    ax[1].set_title("accuracy")
    ax[1].set(xlabel='n_epoch', ylabel='accuracy')
    
    # ax[1].plot(val_accuracy, label = 'val acc')
    ax[1].plot(test_accuracy, label = 'test acc')
    
    ax[1].legend()

    
    set = 'train'
    set_loss = summary[set]['total']
    set_cls_loss = summary[set]['classification']
    set_rec_loss = summary[set]['reconstruction']
    
    ax[2].set_title(f'{set} loss')
    
    ax[2].plot(set_loss, label = 'total')
    # ax[2].plot(set_cls_loss, label = 'cls')
    # ax[2].plot(set_rec_loss, label = 'rec')
    
    ax[2].legend()
    
    # ax[0].set_xlim([50, None])  # This will start the y-axis at 50 and let the y-axis maximum be determined by the data
    # ax[1].set_xlim([50, None]) 
    # ax[2].set_xlim([50, None])
    # ax[2].set_ylim([None, 0.4])
    
    plt.subplots_adjust(wspace=1, hspace=1)
    plt.savefig (output_path)


def get_random_batch(dataloader):
    for batch in dataloader:
        return batch


def save_stacked_img(batch, model, device, save_path):
    
    samples, canny_samples, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
    predicted_labels, reconstructed_canny = model(samples)

    concatenated_images = []

    batch_size = samples.size(0)
    for idx in range(batch_size):
        
        # Ensure all tensors are 4D (batch_size, channels, height, width)
        single_sample = samples[idx].unsqueeze(0)  # Adds the batch dimension
        single_canny = canny_samples[idx].unsqueeze(0)
        single_reconstructed = reconstructed_canny[idx].unsqueeze(0)

      
        # Repeat single-channel tensors along the channel dimension to get 3 channels
        canny_repeated = single_canny.repeat(1, 3, 1, 1)
        reconstructed_repeated = single_reconstructed.repeat(1, 3, 1, 1)
        
        # Concatenate the tensors along the width dimension
        concatenated = torch.cat((single_sample, canny_repeated, reconstructed_repeated), dim=3)
        
        concatenated_images.append(concatenated[0])

    # Convert the list of tensors back to a tensor
    concatenated_images_tensor = torch.stack(concatenated_images)

    # Use save_image with nrow set to the batch size to save a single image with each row containing original, canny, and reconstructed
    torchvision.utils.save_image(concatenated_images_tensor, save_path, nrow=1)    

def draw_rec (model_path, config_path, data_dir, out_dir):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = config_loader(config_path)
    
    domains = config['domains']
    test_domain = config['test_domain']
    reconstruction = config['reconstruction']
    lmda_value = config['lmda_value']
    backbone = config['backbone']

    if not reconstruction:
        lmda_value = 0

    train_domains = [x for x in domains if x != test_domain]

    #Print everything
    print("device: ", str(device))
    print("train domains: ", str(train_domains))
    print("test domain: ", str(test_domain))
    print('lmda value: ', str(lmda_value))
    
    image_datasets = {x: DGDataset(os.path.join(data_dir, x), apply_transform=True, reconstruction=reconstruction)
                      for x in domains}

    train_loader, valid_loader, test_loader, classes = create_data_loaders(image_datasets, train_domains, test_domain,
                                                                  batch_size=4, num_workers=1)
    num_class = len(classes)

    
    model = ResNetAutoEncoderClassifier(backbone, num_class=num_class, reconstruction=reconstruction, state_dict_path=model_path, feature_stylization=False, p=0.0)
    model.to(device)
    
    train_batch = get_random_batch(train_loader)
    val_batch = get_random_batch(valid_loader)
    test_batch = get_random_batch(test_loader)
    
    save_stacked_img(train_batch, model, device, save_path=os.path.join(out_dir, 'train.jpg'))
    save_stacked_img(val_batch, model, device, save_path=os.path.join(out_dir, 'val.jpg'))
    save_stacked_img(test_batch, model, device, save_path=os.path.join(out_dir, 'test.jpg'))
    
    


data_dir = '/export/livia/home/vision/Mcheraghalikhani/datasets/domain_net'
config_path = './configs/resnet-50/domainnet/clipart/config_p_0.1_lmda_1.0.json'

file_name ='5'

model_path = f'./output/{file_name}.pt'
summary_path = f'./output/{file_name}.json'
output_path =f'./output/{file_name}.jpg'

draw_summary(summary_path=summary_path, output_path=output_path)

# draw_rec(model_path=model_path, config_path=config_path, data_dir=data_dir, out_dir='./output')