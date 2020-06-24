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
            U.evaluate_params(f['params'], locals())
            self.loss_f.append(U.load_class(f['module'], f['class'], f['params'], pass_params_as_dict=True))
        self.metrics_f = []
        for f in metrics_f:
            U.evaluate_params(f['params'], locals())
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

    def update_batch_loss(self, data, aggregate = "mean"):
        """Mimics the update_batch but operates on the loss function not the metric function. 
        Logis is same as call but maintains a global state across batches.
        Assumes the loss function either has shape [batch, probs by class or real value for regression] for loss per example or [batch,seq_len, probs]
        for when prediction is for a sequence (lm / conversation / multi-class) in which case it sums over dimention 1."""

        floss = [0] * len(self.loss_f) # loss across all losses in batch
        batch_items = [0] * len(self.loss_f) #total tokens in batch
        #Iterate through all outputs / losses
        if aggregate not in ("mean"):
            raise AttributeError("Expecting aggregate attribute to be mean or sum, got {}".format(aggregate))

        if aggregate == "mean":
            for k, f in enumerate(self.loss_f):
                # for each loss, multiply with the mask
                # Do proper casting based on f_func instance type
                tmp_loss = self._applyloss(f, data['out'][k], data['label'][k])
                self.logger.debug(f"Evaluator tmp loss{k}: {tmp_loss}")
                b_size = data['label'][k].shape[0] # We consider each example to count as 1 (class case). Will override if it's a sequence
                self.logger.debug(f"Batch size: {b_size}")
                if tmp_loss.dim() > 1:
                    # label is a sequence of labels not a single label
                    tmp_loss = tmp_loss.sum(dim=1) #sum over the sequence length resulting in [batch_num,]
                    tmp_loss = (tmp_loss * data['mask'][k])

                    num_items = (data['label'][k] > 0).type(torch.DoubleTensor).to(self.device).sum(dim=1) #Override num_items to be actual tokens TODO: replace 0 with ignore index
                    self.logger.debug(f"Number of tokens in all sequences in batch: {num_items}")

                    #tmp_loss = tmp_loss * num_items # weight each example by it's total tokens.  Shape: [batch_size, }

                    num_items = (num_items * (data['mask'][k] > 0)).sum()
                    self.logger.debug(f"Number of tokens after multiplying with mask: {num_items}")
                else:
                    tmp_loss = (tmp_loss * data['mask'][k])

                    num_items = torch.tensor(1).type(torch.DoubleTensor).to(self.device) # Assume each example is 1. will be broadcasted across batch_size
                    self.logger.debug(f"Number of tokens in all sequences in batch: {num_items}")
                    num_items = (num_items * (data['mask'][k] > 0)).sum()
                    self.logger.debug(f"Number of tokens after multiplying with mask: {num_items}")
                    assert num_items == (data['mask'][k] > 0).sum()
                    #tmp_loss = tmp_loss * num_items


                #tmp_loss = (tmp_loss * data['mask'][k])
                #num_items = (num_items * (data['mask'][k] > 0)).sum()
                #self.logger.debug(f"Number of tokens after multiplying with mask: {num_items}")
                num_items = num_items.data.cpu().numpy()
                if num_items == 0:
                    tmp_loss = 0
                else:
                    tmp_loss = tmp_loss.sum().data.cpu().numpy() #/ num_items
                    #tmp_loss /= b_size
                floss[k] += tmp_loss #mean should be updated to sum / none and other
                batch_items[k] += num_items

                self.logger.debug("Evaluator sum loss across losses: {}".format(floss))
                self.logger.debug("Evaluator Batch total items across losses: {}".format(batch_items))

                self.num_items_loss[k] += batch_items[k] #Add one batch count #data['inp'][0].shape[0]
                self.logger.debug("Evaluator total ruunning sum of items across batches: {}".format(self.num_items_loss))

                # need to calculate sum weighted by total items in batch
                self.current_loss[k] = self.current_loss[k] + (floss[k]) 

            # Sum of all loss across batches
            #self.current_loss = loss

            # Return average loss to this point
            return [f/i for f, i in zip(self.current_loss, self.num_items_loss)]


    def __call__(self, data, aggregate = "mean"):
        """Called during training step to get a single loss value and backpropagate"""
        floss = 0
        batch_items = 0
        #Iterate through all outputs / losses
        if aggregate not in ("mean"):
            raise AttributeError("Expecting aggregate attribute to be mean or sum, got {}".format(aggregate))
        for k, f in enumerate(self.loss_f):
            # for each loss, multiply with the mask
            # Do proper casting based on f_func instance type
            tmp_loss = self._applyloss(f, data['out'][k], data['label'][k])
            self.logger.debug(f"Evaluator tmp loss{k}: {tmp_loss}")
            b_size = data['label'][k].shape[0] # We consider each example to count as 1 (class case). Will override if it's a sequence
            self.logger.debug(f"Batch size: {b_size}")
            if tmp_loss.dim() > 1:
                # label is a sequence of labels not a single label
                tmp_loss = tmp_loss.sum(dim=1) #sum over the sequence length resulting in [batch_num,]
                tmp_loss = (tmp_loss * data['mask'][k])

                num_items = (data['label'][k] > 0).type(torch.DoubleTensor).to(self.device).sum(dim=1) #Override num_items to be actual tokens TODO: replace 0 with ignore index
                self.logger.debug(f"Number of tokens in all sequences in batch: {num_items}")

                #tmp_loss = tmp_loss * num_items # weight each example by it's total tokens.  Shape: [batch_size, }
                num_items = (num_items * (data['mask'][k] > 0)).sum()
                self.logger.debug(f"Number of tokens after multiplying with mask: {num_items}")
            else:
                tmp_loss = (tmp_loss * data['mask'][k])

                num_items = torch.tensor(1).type(torch.DoubleTensor).to(self.device) # Assume each example is 1. will be broadcasted across batch_size
                self.logger.debug(f"Number of tokens in all sequences in batch: {num_items}")
                num_items = (num_items * (data['mask'][k] > 0)).sum()
                self.logger.debug(f"Number of tokens after multiplying with mask: {num_items}")
                assert num_items == (data['mask'][k] > 0).sum()
                #tmp_loss = tmp_loss * num_items


            #tmp_loss = (tmp_loss * data['mask'][k])
            #num_items = (num_items * (data['mask'][k] > 0)).sum()
            #self.logger.debug(f"Number of tokens after multiplying with mask: {num_items}")
            #num_items = num_items
            tmp_loss = tmp_loss.sum() #/ num_items
            floss += tmp_loss #mean should be updated to sum / none and other
            batch_items += num_items
        return floss / batch_items

        for k, f in enumerate(self.loss_f):
            # for each loss, multiply with the mask
            # Do proper casting based on f_func instance type
            tmp_loss = self._applyloss(f, data['out'][k], data['label'][k])
            self.logger.debug(f"Evaluator - labels[{k}]: {data['label'][k]}")
            self.logger.debug(f"Evaluator - output[{k}]: {data['out'][k]}")
            #tmp_loss = f(preds[0][k], y[k].squeeze().type(torch.LongTensor))
            self.logger.debug(f"Evaluator tmp loss{k}: {tmp_loss}")
            num_items = data['label'][k].shape[0] # We consider each example to count as 1 (class case). Will override if it's a sequence
            if tmp_loss.dim() > 1:
                tmp_loss = tmp_loss.sum(dim=1) #sum over the sequence length resulting in [batch_num,]
                if aggregate == "mean":
                    num_items = (data['label'][k] > 0).type(torch.DoubleTensor).sum(dim=1)
                    tmp_loss = tmp_loss * num_items
                self.logger.debug(f"Evaluator tmp loss {k} after summation: {tmp_loss}")
            tmp_loss = (tmp_loss * data['mask'][k])
            if aggregate == "mean":
                num_items = (num_items * (data['mask'][k] > 0)).sum()
                self.logger.debug(f"Number of items after masking: {num_items}")
                if num_items == 0:
                    tmp_loss = 0
                else:
                    tmp_loss = tmp_loss.sum() / num_items
            if aggregate == "sum":
                tmp_loss = tmp_loss.sum()
            floss += tmp_loss#mean should be updated to sum / none and other

        self.logger.debug("Evaluator loss: {}".format(floss))

        return floss

    def _applyloss(self, f, output, label):
        """Calls the loss function with no aggregation.  Should return either [batch_size,] for class or [batch_size, seq_len] for sequence classes"""
        if isinstance(f, torch.nn.CrossEntropyLoss):
            if self.device == 'cuda':
                tmp_loss = f(output, label.squeeze().type(torch.cuda.LongTensor))
            else:
                tmp_loss = f(output, label.squeeze().type(torch.LongTensor))
            return tmp_loss


    def reset(self):
        # Initialization for metrics
        self.num_items = 0
        self.current_metrics = self.get_initial_loss()

        # Initialization for losses
        self.num_items_loss = [0] * len(self.loss_f) 
        self.current_loss = [0] * len(self.loss_f)


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
