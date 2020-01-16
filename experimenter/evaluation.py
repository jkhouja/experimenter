import torch
from experimenter.utils import utils as U
import numpy as np


class ListEvaluator:
    
    def __init__(self, config):
        loss_f = config['evaluator']['params']['loss_f']
        self.loss_f = []
        for f in loss_f:
            self.loss_f.append(U.load_class(f['module'], f['class'], f['params'], pass_params_as_dict=True))
        #self.loss_f = loss_f
        self.reset()

    def update_batch(self, preds, data):
        y = data[1]

        floss = 0
        #Iterate through all outputs / losses
        for k, f in enumerate(self.loss_f):
            # for each loss, multiply with the mask
            # Do proper casting based on f_func instance type
            tmp_loss = f(preds[0][k], y[k].squeeze().type(torch.LongTensor))
            floss += (tmp_loss * data[2][k]).sum()
        
        loss = (self.current_loss * self.num_items) + (floss)
        self.num_items += data[0][0].shape[0]
        self.current_loss = loss / self.num_items
        return self.current_loss


        #Iterate through all outputs / losses
        #for k, f in enumerate(self.loss_f):
        #    # for each loss, multiply with the mask
        #    # Do proper casting based on f_func instance type
        #    tmp_loss = f(preds[k], y[k].squeeze().type(torch.LongTensor))
        #    floss += tmp_loss * b[2][k]
        #
        #val_loss = (val_loss * self.num_items) + (floss)
        #self.num_items += data[0][0].shape[0]
        #val_loss /= self.num_items
        
        # Get score
        #corrects = (preds[0].argmax(1) == y[0].squeeze()).sum().float()
        #total += y[0].shape[0]
        #val_score += corrects

        #val_loss /= j+1 
        #val_score /=  total

    def __call__(self, preds, data):
        floss = 0
        y = data[1]
        #Iterate through all outputs / losses
        for k, f in enumerate(self.loss_f):
            # for each loss, multiply with the mask
            # Do proper casting based on f_func instance type
            tmp_loss = f(preds[0][k], y[k].squeeze().type(torch.LongTensor))
            floss += (tmp_loss * data[2][k]).mean() #mean should be updated to sum / none and other

        return floss


    def reset(self):
        self.num_items = 0
        self.current_loss = self.get_initial_loss()


    def get_initial_loss(self):
        return 0

    def get_max_loss(self):
        return np.inf


    def isbetter(self, a, b):
        return a < b

