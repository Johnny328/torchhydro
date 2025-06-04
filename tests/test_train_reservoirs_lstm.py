
from torchhydro.configs.config import update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
from torchhydro.trainers.trainer import evaluate_model


def test_train_evaluate(reservoir_regulation_lstm_args, config_data):
    update_cfg(config_data, reservoir_regulation_lstm_args)
    train_and_evaluate(config_data)

    # evaluate_model(config_data)
