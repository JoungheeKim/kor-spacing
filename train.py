import logging
import os
import pickle
from torch.utils.data import DataLoader
from my_utills import build_parser, train_decorator
import torch

from loader import load_model, load_dataloader, load_trainer, load_tokenizer

@train_decorator
def run(config):

    assert os.path.isfile(config.data_path), '[{}] 파일이 없습니다.'.format(config.data_path)

    logging.info("##################### Start Training")
    logging.debug(vars(config))

    logging.info("##################### Build Tokenizer")
    tokenizer = load_tokenizer(config)
    ## Tokenizer param Setting
    config.vocab_size = tokenizer.vocab_size
    config.tag_size = tokenizer.tag_size
    config.pad_token_id = tokenizer.pad_token_id

    ##load data loader
    logging.info("##################### Load DataLoader")
    loader = load_dataloader(config, tokenizer)

    config.batch_size = int(config.batch_size / config.gradient_accumulation_steps)
    logging.info("##################### adjusted batch size {}".format(config.batch_size))

    train, valid = loader.get_train_valid_dataset()
    logging.info("##################### Train Dataset size : [" + str(len(train)) + "]")
    logging.info("##################### Valid Dataset size : [" + str(len(valid)) + "]")
    train = DataLoader(train, batch_size=config.batch_size, shuffle=True)
    valid = DataLoader(valid, batch_size=config.batch_size, shuffle=False)

    logging.info("##################### Load Model")
    model = load_model(config, tokenizer)
    model = torch.nn.DataParallel(model)
    model.to(config.device)

    logging.info("##################### Load Trainer")
    trainer = load_trainer(config, model)
    
    ## Training
    logging.info("##################### Training..........")
    best_loss = trainer.train(train, valid)
    logging.info("##################### Best Training Loss : " + str(best_loss))

    ## Testing
    test = loader.get_test_dataset()
    logging.info("##################### Test Dataset size : [" + str(len(test)) + "]")
    test = DataLoader(test, batch_size=config.batch_size, shuffle=False)

    logging.info("##################### Testing..........")
    f1_score = trainer.test(test)
    logging.info("##################### Best Test f1_score : " + str(f1_score))

    result = [config.save_path, best_loss, f1_score]

    return result


if __name__ == "__main__":
    ##load config files
    config = build_parser()
    run(config)