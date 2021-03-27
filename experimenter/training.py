# import cipher_take_home
import datetime
import logging
import os
import pprint
import random
from pathlib import Path

import torch

from experimenter.utils import utils as U


class BasicTrainer:
    """Training class that support cpu and gpu training"""

    def __init__(self, config: dict):
        """Initializes training class and all its submodules from config

        Args:
            config: The configuration dictionary.  For details see sample_config.json

        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.config = config
        self.config["results"] = {}
        self.config["results"]["during_training"] = {}
        self.current_epoch = 0
        self.epochs = config["epochs"]
        self.sample_limit = config.get("sample_limit") or 10
        config["out_path"] = os.path.join(
            config["root_path"],
            "results",
            config["experiment_name"],
            "_".join(
                (
                    datetime.datetime.now().strftime("%b_%d_%Y_%H_%M_%S_%f"),
                    f"{random.random() * 10000:4.0f}".strip(),
                )
            ),
        )
        config["data_path"] = os.path.join(config["root_path"], config["data_subdir"])
        self.out_path = os.path.join(
            config["out_path"], config["experiment_output_file"]
        )
        self.data_path = config["data_path"]
        Path(config["out_path"]).mkdir(parents=True, exist_ok=True)
        self.logger.info(f"All results will be saved in {config['out_path']}")
        self.batch_size = config["processor"]["params"]["batch_size"]
        self.clip = config["grad_clip"] if "grad_clip" in self.config.keys() else 0.25
        # Set up GPU / CPU
        self.gpu_mode = not config["disable_gpu"] and torch.cuda.is_available()
        if self.gpu_mode:
            self.device = "cuda"
        else:
            self.device = "cpu"
        config["device"] = self.device
        self.logger.info("Will be using: {}".format(self.device))

        self.processor = U.load_class(
            config["processor"]["module"], config["processor"]["class"], config
        )

        U.evaluate_params(config["model"]["params"], locals())
        self.model = U.load_class(
            config["model"]["module"], config["model"]["class"], config
        )

        self.evaluator = U.load_class(
            config["evaluator"]["module"], config["evaluator"]["class"], config
        )

        U.evaluate_params(config["optimizer"]["params"], locals())
        self.optimizer = U.load_class(
            config["optimizer"]["module"],
            config["optimizer"]["class"],
            config["optimizer"]["params"],
            pass_params_as_dict=True,
        )

    def predict(self, data: list, decode: bool = True) -> list:
        """Given raw data (unprocessed), run prediction pipeline and return predictions

        Args:
            data: Data raw. Following the expected format fof the task
            decode: If model output needs to be decoded or not, default: True

        Returns:
            res: Result of running the pipeline.
        """
        assert len(data) > 1
        data_as_batches = self.processor(data, list_input=True, as_batches=True)
        # res = []
        # for i, batch in enumerate(data_as_batches):
        #    #res.extend(self.model(U.move_batch(batch, self.device)))
        #    res.extend(self.processor._from_batch(self.model(U.move_batch(batch, self.device))))
        res = self.processor._from_batches(
            [self.model(U.move_batch(batch, self.device)) for batch in data_as_batches]
        )
        # print(res['inp'])
        # print(res['label'])
        # print(res['pred'])
        if decode:
            res = self.processor.decode(res, list_input=True)
        return res

    def evaluate(self, data: list) -> list:
        """Runs the evaluation (metrics not loss - note/ I'm not sure which ) on raw data

        Args:
            data: list of raw data to be evaluated

        Returns:
            metric: The evaluation metric(s)
        """
        assert len(data) > 1
        data_as_batches = self.processor(data, list_input=True, as_batches=True)
        return self._evaluate_batches(data_as_batches)

    def _evaluate_batches(self, data_as_batches):
        self.evaluator.reset()
        val_loss = None
        for j, b in enumerate(data_as_batches):
            # self.logger.debug(b)
            res = self.model(U.move_batch(b, self.device))
            val_loss = self.evaluator.update_batch_loss(res)
        if val_loss is None:
            raise ValueError(
                "Iteration over batches did not happen.\
                        Probably becuase batch_size is larger than the split and drop last is true"
            )
        return val_loss

    def train_model(self, data: list = None) -> dict:
        """Train the model on the data.

        Args:
            data: the data to be trained on. If not provided, default training split will be used

        Returns:
            resulting config file after training
        """

        def log(log_time, total_time, epoch, iteration, total_train_loss, prefix=""):

            val_loss = None
            self.logger.debug("Gradient Norms of last training batch after clipping")
            for p in list(
                filter(lambda p: p.grad is not None, self.model.parameters())
            ):
                self.logger.debug(p.grad.data.norm(2).item())
            if not val_batches:
                self.logger.info(
                    prefix
                    + "Epoch: {}: {}/{} Train duration(s): {:.1f}\
                            \t Train loss (~avg over batches): {:.4f}".format(
                        epoch,
                        iteration * self.batch_size,
                        len(train_batches) * self.batch_size,
                        total_time,
                        total_train_loss,
                    )
                )
                results["during_training"][str(i)] = {
                    "train_loss": float(total_train_loss)
                }
                self.model.eval()
                # Predict sample on train
                # train_res = self.predict(self.processor.get_data(raw=True)[0][:self.sample_limit])
                train_res = self.predict(
                    self.processor.get_sample(
                        indx=0, size=self.sample_limit, split=0, raw=True
                    )
                )
                self.processor.save_split(
                    train_res, f"train_prediction_{epoch}_{iteration}"
                )
            else:
                self.logger.debug(f"Length of val_batches: {len(val_batches)}")
                self.model.eval()
                # Predict sample on train
                # train_res = self.predict(self.processor.get_data(raw=True)[0][:self.sample_limit])
                train_res = self.predict(
                    self.processor.get_sample(
                        indx=0, size=self.sample_limit, split=0, raw=True
                    )
                )
                self.processor.save_split(
                    train_res, f"train_prediction_{epoch}_{iteration}"
                )
                # Predict sample on dev
                # dev_res = self.predict(self.processor.get_data(raw=True)[1][:self.sample_limit])
                dev_res = self.predict(
                    self.processor.get_sample(
                        indx=0, size=self.sample_limit, split=1, raw=True
                    )
                )
                self.processor.save_split(
                    dev_res, f"dev_prediction_{epoch}_{iteration}"
                )

                # Evaluate
                val_loss = self._evaluate_batches(val_batches)
                # train_total_loss = self._evaluate_batches(train_batches)
                # self.logger.info(f"TRAIN LOSS as batches: {train_total_loss}")
                val_loss_str = ",".join(
                    "{:.4f}".format(v) for v in val_loss
                )  # Need to move to evaluator
                results["during_training"][str(i)] = {
                    "train_loss": float(total_train_loss),
                    "val_loss": val_loss_str,
                }

                saved_model = ""
                if self.evaluator.isbetter(val_loss, self.best_loss, is_metric=False):
                    self.best_loss = val_loss
                    # Found best model, save
                    self.model.save()
                    results["best"] = results["during_training"][str(i)]
                    saved_model = " Best model saved"
                    # self.logger.info("Best model saved at: {}".format(config['out_path']))

                self.logger.info(
                    prefix + "Epoch: {}: {}/{} Train duration(s): {:.1f}"
                    " \t Train loss (~avg over batches):"
                    "{:.4f}, validation loss: {} | {}".format(
                        epoch,
                        iteration * self.batch_size,
                        len(train_batches) * self.batch_size,
                        total_time,
                        total_train_loss,
                        val_loss_str,
                        saved_model,
                    )
                )

                # self.logger.info("")
                # Save config
                with open(self.out_path, "w") as f:
                    f.write(U.safe_serialize(config))

        # Generate and process data
        config = self.config
        config_pretty = pprint.pformat(config)
        self.logger.info("Configurations:  " + "-" * 50)
        self.logger.info(config_pretty)
        self.logger.info("--" * 50)

        # train_raw = None
        val_batches = None
        test_batches = None

        if not data:
            data = self.processor.get_data()
            # train_raw = self.processor.get_data(raw=True)[0]

        train_batches = data[0]

        if len(data) > 1:
            val_batches = data[1]
            # val_batches_raw = self.processor.get_data(raw=True)[1]
        if len(data) > 2:
            test_batches = data[2]

        self.logger.info(f"Sample Raw example from train: ")
        self.logger.info(self.processor.get_sample(0))
        self.logger.info(f"After Encoding:")
        sample_ = self.processor.encode(self.processor.get_sample(0))
        self.logger.info(sample_)
        self.logger.info(f"After decoding:")
        sample_ = self.processor.decode(sample_)
        self.logger.info(sample_)
        self.logger.info("--" * 50)
        # qz1 Start training
        self.best_loss = self.evaluator.get_worst_loss()
        results = config["results"]
        self.logger.info(
            f"Training Size: {len(train_batches) * self.batch_size} examples."
        )
        self.logger.info("==" * 50)
        self.logger.info(" " * 20 + "Starting of Training")
        self.logger.info("==" * 50)
        for i in range(self.current_epoch + 1, self.current_epoch + self.epochs + 1):
            self.model.train()  # Set model to train mode
            total_train_loss = 0
            total_train_batches = 0
            # current_epoch = i  # needed to track evaluation?
            start = datetime.datetime.now()
            iteration_beg = datetime.datetime.now()
            for b_num, b in enumerate(train_batches):
                # self.logger.debug(b)
                total_train_batches += 1
                self.model.zero_grad()
                preds = self.model(U.move_batch(b, self.device))
                tloss = self.evaluator(preds)
                total_train_loss += tloss
                tloss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clip, norm_type=2
                )
                self.optimizer.step()

                if b_num * self.batch_size % config["log_interval"] == 0:
                    iteration_done = datetime.datetime.now()
                    iter_total_time = (iteration_done - iteration_beg).total_seconds()
                    self.model.eval()
                    log(
                        datetime.datetime.now().strftime("%m-%d %H:%M"),
                        iter_total_time,
                        i,
                        b_num + 1,
                        total_train_loss / total_train_batches,
                    )
                    iteration_beg = datetime.datetime.now()
                    self.model.train()  # Set model to train mode

            done = datetime.datetime.now()
            epoch_time = (done - start).total_seconds()

            # Need to dynamically decide if avg of loss is needed or sum
            # This will break when loss is changed between mean / sum.
            # Might be slightly off if last batch is not dropped and has size != to batch_size
            total_train_loss /= total_train_batches
            self.logger.info("-" * 50)
            self.model.eval()
            log(
                done.strftime("%m-%d %H:%M"),
                epoch_time,
                i,
                b_num + 1,
                total_train_loss,
                prefix="END OF EPOCH | ",
            )
            self.logger.info("-" * 50)
            self.model.train()

        self.logger.info("==" * 50)
        self.logger.info(" " * 20 + "End of Training")
        self.logger.info("==" * 50)

        if test_batches is not None:
            self.model.eval()  # Set model to train mode
            test_val = self._evaluate_batches(test_batches)
            test_loss_str = ",".join(
                str(v) for v in test_val
            )  # Need to move to evaluator
            results["test"] = {test_loss_str}
            self.logger.info("Test metrics: {}".format(test_loss_str))
            # Save config
            with open(self.out_path, "w") as f:
                f.write(U.safe_serialize(config))

        config["results"] = results
        return config

    def __call__(self):
        if self.config["do_train"]:
            self.logger.info("Training flag is set to true. Will train")
            _ = self.train_model()
            self.logger.info("Training completed")

        if self.config["do_predict"]:
            self.logger.info(
                "Predict flag is set to true. Will run prediction on all splits using best model"
            )
            # load model
            self.model.load()
            self.model.eval()  # Set model to train mode
            data = self.processor.get_data(raw=True)
            train = self.predict(data[0])
            self.logger.info("Prediction finished on train")
            self.processor.save_split(train, "train_prediction")
            if len(data) > 1:
                dev = self.predict(data[1])
                self.processor.save_split(dev, "dev_prediction")
                self.logger.info("Prediction finished on dev")
            if len(data) > 2:
                test = self.predict(data[2])
                self.processor.save_split(test, "test_prediction")
                self.logger.info("Prediction finished on test")
