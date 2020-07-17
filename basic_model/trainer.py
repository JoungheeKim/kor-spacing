import os
import re
import torch
from tqdm import tqdm
import torch.optim as optim
import logging
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import json
import numpy as np
try:
    from apex import amp
except ModuleNotFoundError:
    print('[Training Process] apex is not available')


class Trainer():
    def __init__(self, model, config, device="cpu"):

        if config.model_name == 'bert':
            optimizer = optim.Adam(
                [
                    {"params": model.module.bert.parameters(), "lr": config.lr_bert},
                    {"params": model.module.hidden2tag.parameters(), "lr": config.lr}
                ])

        else:
            optimizer = optim.Adam(model.parameters(), lr=config.lr)

        if config.use_apex:
            if config.opt_level == 'O2':
                model, optimizer = amp.initialize(
                    model, optimizer, opt_level=config.opt_level,
                    keep_batchnorm_fp32=True, loss_scale="dynamic"
                )
            else:
                model, optimizer = amp.initialize(
                    model, optimizer, opt_level=config.opt_level,
                    keep_batchnorm_fp32=None, loss_scale="dynamic"
                )

        self.model = model
        self.optimizer = optimizer

        self.config = config
        self.device = device
        self.use_apex = config.use_apex
        self.clip_grad = config.clip_grad

        self.save_path = os.path.join(config.save_path)
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)

        device = config.device
        config.device = "gpu"
        with open(os.path.join(self.save_path, "config.json"), 'w') as outfile:
            json.dump(vars(config), outfile)
        config.device = device

        super().__init__()

        self.n_epochs = config.n_epochs
        self.lower_is_better = True
        self.best = {'epoch': 0,
                     'config': config
                     }
        logging.info("##################### Init Trainer")


    def get_best_model(self):
        if 'model' in self.best:
            self.model.load_state_dict(self.best['model'])
        return

    def save_training(self, path):
        torch.save(self.best, path)

    def train(self, train, valid):
        '''
        Train with given train and valid iterator until n_epochs.
        If early_stop is set,
        early stopping will be executed if the requirement is satisfied.
        '''


        logging.info("run train")
        best_loss = float('Inf') * (1 if self.lower_is_better else -1)
        lowest_after = 0

        progress_bar = tqdm(range(self.best['epoch'], self.n_epochs),
                            desc='Training: ',
                            unit='epoch'
                            )


        for idx in progress_bar:  # Iterate from 1 to n_epochs
            avg_train_loss = self.train_epoch(train)
            avg_valid_loss = self.validate_epoch(valid)
            progress_bar.set_postfix_str('train_loss=%.4e valid_loss=%.4e min_valid_loss=%.4e' % (avg_train_loss,
                                                                                                  avg_valid_loss,
                                                                                                  best_loss))
            logging.debug('train_loss=%.4e valid_loss=%.4e min_valid_loss=%.4e' % (avg_train_loss,
                                                                                   avg_valid_loss,
                                                                                   best_loss))
            if (self.lower_is_better and avg_valid_loss < best_loss) or \
               (not self.lower_is_better and avg_valid_loss > best_loss):
                # Update if there is an improvement.
                best_loss = avg_valid_loss
                lowest_after = 0

                self.best['model'] = self.model.state_dict()
                self.best['epoch'] = idx + 1

                # Set a filename for model of last epoch.
                # We need to put every information to filename, as much as possible.

                model_name = "best.pwf"
                self.save_training(os.path.join(self.save_path, model_name))
            else:
                lowest_after += 1

                if lowest_after >= self.config.early_stop and \
                   self.config.early_stop > 0:
                    logging.debug("early stop")
                    progress_bar.close()
                    return best_loss
        progress_bar.close()
        return best_loss

    def train_epoch(self, train):
        '''
        Train an epoch with given train iterator and optimizer.
        '''
        total_loss, total_count = 0, 0
        avg_loss = 0


        progress_bar = tqdm(train,
                            desc='Training: ',
                            unit='batch'
                            )
        # Iterate whole train-set.
        self.optimizer.zero_grad()
        for idx, batch in enumerate(progress_bar):
            batch = tuple(t.to(self.device) for t in batch)
            token_ids, lenghts, labels = batch

            loss = self.model(token_ids, lenghts, labels)

            ## mean is needed because Data parrell function
            loss = torch.mean(loss)

            # Simple math to show stats.
            total_loss += float(loss)
            avg_loss = total_loss / (idx + 1)

            progress_bar.set_postfix_str('avg_loss=%.4e' % (avg_loss))

            if self.config.gradient_accumulation_steps > 1:
                loss = loss / self.config.gradient_accumulation_steps

            # Calcuate loss and gradients with back-propagation.
            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Take a step of gradient descent.
                if self.config.use_apex:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.config.clip_grad)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad)

                self.optimizer.step()
                self.optimizer.zero_grad()

        progress_bar.close()

        return avg_loss

    def validate_epoch(self, valid):
        total_loss, total_count = 0, 0
        avg_loss = 0

        self.model.eval()
        with torch.no_grad():
            progress_bar = tqdm(valid, desc='Validation: ', unit='batch')
            # Iterate for whole valid-set.
            for idx, batch in enumerate(progress_bar):
                batch = tuple(t.to(self.device) for t in batch)
                token_ids, lenghts, labels = batch
                loss = self.model(token_ids, lenghts, labels)

                ## mean is needed because Data parrell function
                loss = torch.mean(loss)

                total_loss += float(loss)
                avg_loss = total_loss / (idx+1)

                progress_bar.set_postfix_str('avg_loss=%.4e' % (avg_loss))

            progress_bar.close()
        self.model.train()
        return avg_loss

    def test(self, data):
        self.get_best_model()

        total_label = []
        total_pred = []

        self.model.eval()
        with torch.no_grad():
            progress_bar = tqdm(data, desc='testing: ', unit='batch')
            # Iterate for whole valid-set.
            for idx, batch in enumerate(progress_bar):
                batch = tuple(t.to(self.device) for t in batch)
                token_ids, lenghts, labels = batch
                label_hat = self.model(token_ids, lenghts)

                ps = torch.exp(label_hat)
                top_p, top_class = ps.topk(1, dim=2)
                pred_info = top_class.squeeze(2).tolist()
                label_info = labels.cpu().tolist()
                length_info = lenghts.cpu().tolist()

                for temp_label, temp_pred, temp_len in zip(label_info, pred_info, length_info):
                    total_label += temp_label[:temp_len]
                    total_pred += temp_pred[:temp_len]

            progress_bar.close()
        self.model.train()

        from sklearn.metrics import f1_score
        f1 = f1_score(total_label, total_pred, average='macro')

        return f1