import torch
from experimenter.utils import utils as U
import numpy as np
import logging


class Accuracy:

    def __call__(self, prediction: torch.Tensor, label: torch.Tensor):
        """Get accuracy of a multi-class classification for a batch
    
        Args:
            prediction: Batch of shape (batch_size, num_classes)
            label: Batch true labels of shape (batch_size) 
        Returns:
    
            score: scalar of accuracy for this bach
        """
        assert isinstance(prediction, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        logging.debug(f'prediction: {prediction}')
        logging.debug(f'true label: {label}')
        corrects = (prediction == label).sum().float()
        total = label.shape[0]
        val_score = corrects
        val_score /=  total
        logging.debug(f'Val_score: {val_score}')
        return val_score.cpu().detach().numpy()

class Dummy:

    def __call__(self, prediction: torch.Tensor, label: torch.Tensor):
        """A dummy class that returns 0 to be used with multi-task outputs that are not of interest
    
        Args:
            prediction: Batch of shape (batch_size, num_classes)
            label: Batch true labels of shape (batch_size) 
        Returns:
    
            score: zero
        """
        return 0

class ListEvaluator:
    
    def __init__(self, config):
        self.logger = logging.getLogger(self.__class__.__name__)
        loss_f = config['evaluator']['params']['loss_f']
        metrics_f = config['evaluator']['params']['metrics_f']
        self.device = config['device']
        self.loss_f = []
        for f in loss_f:
            self.loss_f.append(U.load_class(f['module'], f['class'], f['params'], pass_params_as_dict=True))
        self.metrics_f = []
        for f in metrics_f:
            self.metrics_f.append(U.load_class(f['module'], f['class'], f['params'], pass_params_as_dict=True))
        self.reset()

    def update_batch(self, data):

        #floss = 0
        res = self.get_initial_loss() 
        for k, f in enumerate(self.metrics_f):
            res[k] += f(data['pred'][k], data['label'][k])

        self.num_items += data['inp'][0].shape[0] #Add one batch count #data['inp'][0].shape[0]
        loss = [current_metric + (metric * data['inp'][0].shape[0]) for current_metric, metric in zip(self.current_metrics, res)]

        self.current_metrics = loss
        return [metric / self.num_items for metric in self.current_metrics]

    def update_batch_loss(self, data, aggregate = "sum"):
        """Mimics the update_batch but operates on the loss function not the metric function"""

        floss = 0
        #Iterate through all outputs / losses
        for k, f in enumerate(self.loss_f):
            # for each loss, multiply with the mask
            # Do proper casting based on f_func instance type

            tmp_loss = self._applyloss(f, data['out'][k], data['label'][k])
            #tmp_loss = f(preds[0][k], y[k].squeeze().type(torch.LongTensor))
            if tmp_loss.dim() > 1:
                tmp_loss = tmp_loss.sum(dim=1) #sum over the sequence length (vocab)
            floss += (tmp_loss * data['mask'][k]) #mean should be updated to sum / none and other

        if aggregate == "mean":
            floss = floss.mean()
        elif aggregate == "sum":
            floss = floss.sum()
        else:
            raise AttributeError("Expecting aggregate attribute to be mean or sum, got {}".format(aggregate))

        self.num_items_loss += data['inp'][0].shape[0] #Add one batch count #data['inp'][0].shape[0]
        loss = self.current_loss + (floss * data['inp'][0].shape[0]) 

        # Sum of all loss across batches
        self.current_loss = loss

        # Return average loss to this point
        return [loss.data.cpu().numpy() / self.num_items_loss]

    def get_metrics(self, data):
        #Iterate through all outputs / losses
        res = []
        for k, f in enumerate(self.metrics_f):
        #    # for each loss, multiply with the mask
        #    # Do proper casting based on f_func instance type
            res.append(f(data['pred'][k], data['label'][k]))
        #    floss += tmp_loss * b[2][k]
        #
        #val_loss = (val_loss * self.num_items) + (floss)
        #self.num_items += data[0][0].shape[0]
        #val_loss /= self.num_items
        return res
        
        # Get score

    def __call__(self, data, aggregate = "mean"):
        floss = 0
        #Iterate through all outputs / losses
        for k, f in enumerate(self.loss_f):
            # for each loss, multiply with the mask
            # Do proper casting based on f_func instance type
            tmp_loss = self._applyloss(f, data['out'][k], data['label'][k])
            self.logger.debug(f"Evaluator - labels[{k}]: {data['label'][k]}")
            self.logger.debug(f"Evaluator - output[{k}]: {data['out'][k]}")
            #tmp_loss = f(preds[0][k], y[k].squeeze().type(torch.LongTensor))
            self.logger.debug(f"Evaluator tmp loss{k}: {tmp_loss}")
            if tmp_loss.dim() > 1:
                tmp_loss = tmp_loss.sum(dim=1) #sum over the sequence length (vocab)
                self.logger.debug(f"Evaluator tmp loss {k} after summation: {tmp_loss}")
            floss += (tmp_loss * data['mask'][k]) #mean should be updated to sum / none and other


        if aggregate == "mean":
            floss = floss.mean()
        elif aggregate == "sum":
            floss = floss.sum()
        else:
            raise AttributeError("Expecting aggregate attribute to be mean or sum, got {}".format(aggregate))
        self.logger.debug("Evaluator loss: {}".format(floss))

        return floss

    def _applyloss(self, f, output, label):
        if isinstance(f, torch.nn.CrossEntropyLoss):
            if self.device == 'cuda':
                tmp_loss = f(output, label.squeeze().type(torch.cuda.LongTensor))
            else:
                tmp_loss = f(output, label.squeeze().type(torch.LongTensor))
            return tmp_loss


    def reset(self):
        self.num_items = 0
        self.current_metrics = self.get_initial_loss()

        self.num_items_loss = 0
        self.current_loss = 0


    def get_initial_loss(self):
        res = [0] * len(self.metrics_f) 
        return res

    def get_worst_metric(self):
        return [0] * len(self.metrics_f)


    def get_worst_loss(self):
        return [np.inf] * len(self.metrics_f)

    def isbetter(self, a, b, is_metric = True):
        """If is metric, we're assuming higher is better (think accuracy), else, it's a loss and lower is better"""
        # Bad implementation, find a way to compare other metrics
        if is_metric:
            return np.all(a > b)
        else:
            return np.all(a < b)
