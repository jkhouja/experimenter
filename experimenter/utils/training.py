#import cipher_take_home
import datetime
import re
from collections import defaultdict
from typing import DefaultDict, Union, List
import json
import numpy as np
import torch
import os
import argparse
import logging
from experimenter.utils import utils as U


     
class BasicTrainer():

    def __init__(self, kwargs):
        config = kwargs
        self.logger = logging.getLogger(self.__class__.__name__)
        self.epochs = config['epochs']
        self.gpu_mode = not config['disable_gpu'] and torch.cuda.is_available()
        config['out_path'] = os.path.join(config['root_path'], config['experiment_name'], datetime.datetime.now().strftime("%b_%d_%Y_%H_%M"))
        self.out_path = os.path.join(config['out_path'], config['experiment_output_file']) 
        from pathlib import Path
        Path(config['out_path']).mkdir(parents=True, exist_ok=True)
        if self.gpu_mode:
            self.device = "cuda"
        else:
            self.device = "cpu"

        config['device'] = self.device
        self.logger.info("Will be using: {}".format(self.device))
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
        self.logger.info("Total params: {}".format(total_params))
        #self.logger.info("Vocab (Size= {}):".format(vocab_size))

    def predict(self, data, decode=True):
        assert len(data) > 1
        data_as_batches = self.processor(data, list_input=True, as_batches=True)
        res = []
        for i, batch in enumerate(data_as_batches):
            #res.extend(self.model(batch))
            res.extend(self.processor._from_batch(self.model(U.move_batch(batch, self.device))))

        if decode:
            res = self.processor.decode(res, list_input=True)
        return res
    
    def evaluate(self, data):
        assert len(data) > 1
        data_as_batches = self.processor(data, list_input=True, as_batches=True)
        return self._evaluate_batches(data_as_batches)

    def _evaluate_batches(self, data_as_batches):
        self.evaluator.reset()
        for j, b in enumerate(data_as_batches):
            res = self.model(U.move_batch(b, self.device))
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
        best_loss = self.evaluator.get_worst_metric()
        results = config['results']
        
        #self.model.to(self.device)
        self.logger.info("Starting training:")
        for i in range(self.current_epoch + 1, self.current_epoch + self.epochs + 1):
            self.current_epoch = i
            start = datetime.datetime.now()
            for b in train_batches:
                
                self.model.zero_grad()
                preds = self.model(U.move_batch(b, self.device)) 
                tloss = self.evaluator(preds)
                tloss.backward()
                self.optimizer.step()

            done = datetime.datetime.now()
            epoch_time = (done - start).total_seconds()
            if i % config['log_interval'] == 0:
                val_loss = None
                if not val_batches:
                    self.logger.info("Epoch: {}: Duration(s): {} Train loss (last batch): {}".format(i, epoch_time, tloss))
                    results['during_training'][str(i)] = {'train_loss': float(tloss.detach())}
                else:
                    val_loss = self._evaluate_batches(val_batches) 
                    val_loss_str = ",".join(str(v) for v in val_loss) # Need to move to evaluator
                    self.logger.info("Epoch: {}: Duration(s): {} Train loss (last batch): {}, validation metrics: {}".format(i, epoch_time, tloss, val_loss_str))
                    results['during_training'][str(i)] = {'train_loss': float(tloss.detach()), 'val_loss': val_loss_str}
    
                    if self.evaluator.isbetter(val_loss, best_loss):
                        best_loss = val_loss
                        # Found best model, save
                        self.model.save()
                        results['best'] = results['during_training'][str(i)]
                        self.logger.info("Best model saved at: {}".format(config['out_path']))

                    # Save config
                    with open(self.out_path, 'w') as f:
                        f.write(U.safe_serialize(config))
        
        if test_batches is not None:
            test_val = self._evaluate_batches(test_batches)
            test_loss_str = ",".join(str(v) for v in test_val) # Need to move to evaluator
            results['test'] = {test_loss_str}
            self.logger.info("Test metrics: {}".format(test_loss_str))
            # Save config
            with open(self.out_path, 'w') as f:
                f.write(U.safe_serialize(config))

        config['results'] = results
        #self.config = config

        return config


