import os
import torch
from argparse import ArgumentParser

from custom_dataloder import DGDataset
from models import ResNetAutoEncoderClassifier
from utils import create_data_loaders, train_model, config_loader


def get_args_parser():

    parser = ArgumentParser('Structure-Aware Feature Stylization for Domain Generalization', add_help=False)

    parser.add_argument("-c", "--config", dest="config_path", metavar="config file", required=True)
    parser.add_argument("-d", "--data_dir", dest="data_dir", metavar="data dir", required=True)

    return parser


def main(config, data_dir):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # data_path = config['data_path']
    domains = config['domains']
    test_domain = config['test_domain']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    num_workers = config['num_workers']
    reconstruction = config['reconstruction']
    feature_stylization = config['feature_stylization']
    save_path = config['save_path']
    lmda_value = config['lmda_value']
    p_value = config['p_value']
    lr = config['lr']
    backbone = config['backbone']


    if not feature_stylization:
        p_value = 0

    if not reconstruction:
        lmda_value = 0

    train_domains = [x for x in domains if x != test_domain]

    #Print everything
    print("train domains: ", str(train_domains))
    print("test domain: ", str(test_domain))
    print('feature_stylization: ', str(feature_stylization))
    print('reconstruction: ', str(reconstruction))
    print('lmda value: ', str(lmda_value))
    print('p value: ', str(p_value))
    print('batch size: ', str(batch_size))
    print('initial lr: ', str(lr))


    # Creating dataloaders
    image_datasets = {x: DGDataset(os.path.join(data_dir, x), apply_transform=True, reconstruction=reconstruction)
                      for x in domains}

    train_loader, valid_loader, test_loader, classes = create_data_loaders(image_datasets, train_domains, test_domain,
                                                                  batch_size=batch_size, num_workers=num_workers)
    num_class = len(classes)

    print(f"num classes {num_class}")
    
    # Preparing a directory to save models and results
    model_save_path = os.path.join(save_path, test_domain,
                                        "lmda_{}_p_{}_lr_{}".format(lmda_value, p_value, lr))

    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)


    # Building the model
    model = ResNetAutoEncoderClassifier(backbone, num_class=num_class, reconstruction=reconstruction,
                                        feature_stylization=feature_stylization, p=p_value)


    # Train the model
    summary = train_model(num_epochs, model, train_loader, valid_loader, test_loader,
                          model_directory=model_save_path,
                          lr=lr,
                          reconstruction=reconstruction,
                          device=device, lmda=lmda_value, model_name="lmda_{}_p_{}_lr_{}".format(lmda_value, p_value, lr))




if __name__ == '__main__':

    args = get_args_parser()
    args = args.parse_args()

    config_path = args.config_path
    data_dir = args.data_dir
    config = config_loader(config_path)

    main(config , data_dir)
