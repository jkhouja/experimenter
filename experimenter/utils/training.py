#import cipher_take_home
import re
from collections import defaultdict
from typing import DefaultDict, Union, List
import json
import numpy as np
import torch
import argparse
from experimenter.utils import utils as U


     
class BasicTrainer():

    def __init__(self, kwargs):
        config = kwargs
        self.epochs = config['epochs']
        self.processor = U.load_class(config['processor']['module'], config['processor']['class'],  config )
        U.evaluate_params(config['model']['params'], locals())
        self.model = U.load_class(config['model']['module'], config['model']['class'], config)
        self.evaluator = U.load_class(config['evaluator']['module'], config['evaluator']['class'],  config )
        U.evaluate_params(config['optimizer']['params'], locals())
        self.optimizer = U.load_class(config['optimizer']['module'], config['optimizer']['class'],config['optimizer']['params'], pass_params_as_dict=True)
        #Needs to be loaded if trainer is laoded 
        self.config = config
        self.config['results'] = {}
        self.config['results']['during_training'] = {}
        self.current_epoch = 0     
        
        # Print statistics
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Total params: {}".format(total_params))
        #print("Vocab (Size= {}):".format(vocab_size))

    def predict(self, data, decode=True):
        assert len(data) > 1
        data_as_batches = self.processor(data, list_input=True, as_batches=True)
        res = []
        for i, batch in enumerate(data_as_batches):
            #res.extend(self.model(batch))
            res.extend(self.processor._from_batch(self.model(batch)))

        if decode:
            res = self.processor.decode(res, list_input=True)
        return res
    
    def evaluate(self, data):
        assert len(data) > 1
        data_as_batches = self.processor(data, list_input=True, as_batches=True, data_type="full")
        return self._evaluate_batches(data_as_batches)

    def _evaluate_batches(self, data_as_batches):
        self.evaluator.reset()
        for j, b in enumerate(data_as_batches):
            res = self.model(b)
            val_loss = self.evaluator.update_batch(res)
        return val_loss
        

    def train_model(self, data=None) -> dict:
        # Generate and process data    
        config = self.config

        if not data:
            data = self.processor.get_data()

        
        train_batches = data[0] 
        val_batches = None
        test_batches = None
        if len(data) > 1:
            val_batches = data[1]
        if len(data) > 2:
            test_batches = data[2]
    
        # Start training
        best_loss = self.evaluator.get_max_loss()
        results = config['results']
        
        print("Starting training:")
        for i in range(self.current_epoch + 1, self.current_epoch + self.epochs + 1):
            self.current_epoch = i
            for b in train_batches:
                
                self.model.zero_grad()
                preds = self.model(b) 
                tloss = self.evaluator(preds)
                tloss.backward()
                self.optimizer.step()
    
            if i % config['log_interval'] == 0:
                val_loss = None
                if not val_batches:
                    print("Epoch: {}: Train loss (last batch): {}".format(i, tloss))
                    results['during_training'][str(i)] = {'train_loss': float(tloss.detach())}
                else:
                    val_loss = self._evaluate_batches(val_batches) 
                    print("Epoch: {}: Train loss (last batch): {}, validation loss: {}".format(i, tloss, val_loss))
                    results['during_training'][str(i)] = {'train_loss': float(tloss.detach()), 'val_loss': float(val_loss.detach())}
    
                    if self.evaluator.isbetter(val_loss, best_loss):
                        best_loss = val_loss
                        # Found best model, save
                        torch.save(self.model.state_dict(), config['model_path'])
                        results['best'] = results['during_training'][str(i)]
                        # Save config
                        #with open(config['experiment_output_path'], 'w') as f:
                        #     json.dump(config, f)
                        print("Best model saved at: {}".format(config['model_path']))
        
        if test_batches is not None:
            test_val = self._evaluate_batches(test_batches)
            results['test'] = {test_val}
            print("Test loss: {}".format(test_val))

        config['results'] = results
        #self.config = config

        return config


