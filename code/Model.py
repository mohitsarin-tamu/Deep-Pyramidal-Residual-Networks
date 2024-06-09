### YOUR CODE HERE
import torch
import os, time
import numpy as np
import torch.nn as nn
from ImageUtils import parse_record
from tqdm import tqdm
import Network as PN
from amp import AMP
import matplotlib.pyplot as plt
import torch.nn.functional as F

"""This script defines the training, validation and testing process.
"""

class MyModel(object):

    def __init__(self, configs):
        
        # define configs parameters
        self.weight_decay = configs['weight_decay']
        self.batch_size = configs['batch_size']
        self.epochs = configs['epochs']
        self.save_interval = configs['save_interval']
        self.modeldir =configs['modeldir']
        self.learning_rate = configs['learning_rate']
        self.milestones = configs['milestones']
        self.gamma = configs['gamma']
        self.clip_norm = configs['clip_norm']
        self.momentum = configs['momentum']
        self.epsilon = configs['epsilon']
        self.inner_lr = configs['inner_lr']
        self.inner_iter = configs['inner_iter']
        self.lrchange_ep = configs['lrchange_ep']
        self.best_model_path =configs['best_model_path']
        
        self.configs = configs
        self.network = PN.PyramidNet(configs).to('cuda')
        # define cross entropy loss and optimizer
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        #self.optimizer = torch.optim.SGD(self.network.parameters(), lr=0.1, weight_decay=self.weight_decay)
        self.optimizer = AMP(params=filter(lambda p: p.requires_grad, self.network.parameters()),
                        lr=self.learning_rate,
                        epsilon=self.epsilon,
                        inner_lr=self.inner_lr,
                        inner_iter=self.inner_iter,
                        base_optimizer=torch.optim.SGD,
                        momentum=self.momentum,
                        weight_decay=self.weight_decay,
                        nesterov=True)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.milestones, self.gamma)
        

    # Model setup parameters are defined in __init__
    #def model_setup(self, configs):
        

    def train(self, x_train, y_train, configs, x_valid=None, y_valid=None):
        self.network.train()
        num_samples = x_train.shape[0]
        num_batches = num_samples // self.batch_size

        train_losses = []
        #print('Training...')
        for epoch in range(1, self.epochs + 1):
            
            start_time = time.time()
            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]
            
            if epoch % self.lrchange_ep  == 0:
                for i in self.optimizer.param_groups:
                    i['lr'] /= 10
            
            #print(num_batches)
            for i in range(num_batches):
                
                #print(num_batches.shape)
                start_index = i * self.batch_size
                end_index = min((i + 1) * self.batch_size, curr_x_train.shape[0])
                #batch_x = [(parse_record(curr_x_train[i], True)) for i in range(start_index,end_index)]
                batch_x = [parse_record(curr_x_train[i], True) for i in range(start_index,end_index)]
                batch_y = curr_y_train[start_index:end_index]
                #print(batch_x[0].shape,batch_x[1].shape )

                
                #batch_x, batch_y = batch_x.#cuda, batch_y.#cuda
                #batch_x = batch_x.double()
                new_batch_x = torch.tensor(np.array(batch_x), dtype=torch.float).to('cuda')
                #print(new_batch_x.shape)
                new_batch_y = torch.tensor(np.array(batch_y), dtype=torch.long).to('cuda')
                #print("Code-Reaching ")
                
                # define closure function for AMP optimizer
                def closure():
                    self.optimizer.zero_grad()
                    #outputs = self.model(inputs)
                    final_output = self.network(new_batch_x).to('cuda')
                    #loss = self.cross_entropy_loss(outputs, targets)
                    loss = self.cross_entropy_loss(final_output,new_batch_y)
                    loss.backward()
                    # using the clip norm parameter as described in original paper
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_norm)
                    return final_output, loss
                
                final_output, loss = self.optimizer.step(closure)
                #self.optimizer.step()
                #print('Done')
                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='\r', flush=True)
            
            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, loss, duration))
            
            train_losses.append(loss) 

            # save after define number of epochs
            if epoch % self.save_interval == 0:
                self.save(epoch)
        
        # plot loss       
        self.plot_loss(train_losses)

    def predict_prob(self, x):
        # take the best model saved weights path here
        bestmodel = os.path.join(self.best_model_path)
        self.load(bestmodel)
        self.network.load_state_dict(torch.load(bestmodel))
        self.network.eval()
        output_val = []
        with torch.no_grad():
            predict_data = torch.tensor(np.array([parse_record(record, False) for record in x]), dtype=torch.float32).to('cuda')
            for i in range(x.shape[0]):
                prediction_prob = self.network(predict_data[i].reshape(1,3,32,32))
                probs = F.softmax(prediction_prob, dim=1)
                # squeeze probs to be in proper shape
                squeezed_probs = probs.squeeze().cpu().numpy()
                output_val.append(squeezed_probs)

            print(output_val)
            output = np.array(output_val)
            print(output.shape)
            return output
    
    def evaluate(self, x, y, checkpoint_num_list):
        self.network.eval()
        print('### Test or Validation ###')
        for checkpoint_num in checkpoint_num_list:
            checkpointfile = os.path.join(self.modeldir, 'model-%d.ckpt'%(checkpoint_num))
            self.load(checkpointfile)

            preds = []
            for i in tqdm(range(x.shape[0])):
                with torch.no_grad():

                    input = parse_record(x[i], training =False).reshape(1,3,32,32)
                    input = torch.tensor(input, dtype=torch.float32).to('cuda')  
                    network_output = self.network(input)  
                    _, result = torch.max(network_output.data, 1) 
                    #print("Result_printing")
                    preds.append(result.item())

            # calculate output and print accuracy
            y = torch.tensor(y)
            preds = torch.tensor(preds)
            print('Test accuracy: {:.4f}'.format(torch.sum(preds==y)/y.shape[0]))
            
    
    def save(self, epoch):
        checkpoint_path = os.path.join(self.modeldir, 'model-%d.ckpt'%(epoch))
        os.makedirs(self.modeldir, exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")
    
    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))
        
    # function to plot the training loss
    def plot_loss(self, train_losses):
        epochs = range(1, len(train_losses) + 1)
        train_losses_tensor = torch.tensor(train_losses)
        train_losses_cpu = train_losses_tensor.cpu().numpy()
        plt.plot(epochs, train_losses_cpu, 'b', label='Training loss')
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        # save the image
        plt.savefig('loss.png') 
        #plt.show()
        plt.close()
    

### END CODE HERE
