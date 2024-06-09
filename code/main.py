### YOUR CODE HERE
## importing required modules
import torch
import os, argparse
import numpy as np
from Model import MyModel
from DataLoader import load_data, train_valid_split, load_testing_images
from Configure import model_configs, training_configs
from ImageUtils import visualize

# define the args
parser = argparse.ArgumentParser()
parser.add_argument("--mode", default='train')
parser.add_argument("--data_dir", help="path to the data", default='../data/private_test_images_2024.npy')
#parser.add_argument("--data_dir", help="path to the data", default='/content/drive/MyDrive/cifar-10-batches-py')
parser.add_argument("--save_dir", default='../data/', help="path to save the results")
parser.add_argument('--data', metavar='DIR', help='path to dataset')
parser.add_argument("--save_interval", type=int, default=10,
                    help='save the checkpoint when epoch MOD save_interval == 0')
parser.add_argument("--modeldir", type=str, default='model', help='model directory')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--depth', default=110, type=int,
                    help='depth of the network (default: 32)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='to use basicblock for CIFAR datasets (default: bottleneck)')
parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                    help='to print the status at every iteration')
parser.add_argument('--alpha', default=300, type=int,
                    help='number of new channel increases per depth (default: 300)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='default:True, that is no augmentation')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument("--result_dir", type=str, default='predictions', help='prediction result directory')

# intializing the bottleneck parameter as true
parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)
parser.set_defaults(augment=True)

args = parser.parse_args()

if __name__ == '__main__':
    model = MyModel(model_configs)

    if args.mode == 'train':
        x_train, y_train, x_test, y_test = load_data(args.data_dir)
        #x_train, y_train, x_valid, y_valid = train_valid_split(x_train, y_train)

        model.train(x_train, y_train, training_configs)
        #model.evaluate(x_valid, y_valid, [120, 150, 200, 250, 260, 270, 280, 290, 300])
        #model.evaluate(x_valid, y_valid, [1,2])
        #model.evaluate(x_test, y_test, [270, 280, 290, 300])

    elif args.mode == 'test':
        # Testing on public testing dataset
        _, _, x_test, y_test = load_data(args.data_dir)
        model.evaluate(x_test, y_test,[150,210,220,230,240,250,250,270,280,290,300])

    elif args.mode == 'predict':
        # Loading private testing dataset
        x_test = load_testing_images(args.data_dir)
        # visualizing the first testing image to check your image shape
        visualize(x_test[0], 'test.png')
        # Predicting and storing results on private testing dataset
        predictions = model.predict_prob(x_test)
        np.save(args.result_dir, predictions)

### END CODE HERE
