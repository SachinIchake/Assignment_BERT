import config
import dataset
import engine
import torch
import pandas as pd
import torch.nn as nn
import numpy as np

from model import BERTBaseUncased
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


def run():
    dfx = pd.read_csv(config.TRAINING_FILE).fillna("none").reset_index(drop=True)
    # df_test = pd.read_csv(config.TESTING_FILE).fillna("none").reset_index(drop=True)

    df_train, df_test = model_selection.train_test_split(
        dfx, test_size=0.1, random_state=42, stratify=dfx.label.values
    )

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    train_dataset = dataset.BERTDataset(
        text=df_train.title.values, label=df_train.label.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    test_dataset = dataset.BERTDataset(
        text=df_test.title.values, label=df_test.label.values
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.TEST_BATCH_SIZE, num_workers=1
    )

    device = torch.device("cpu")
    model = BERTBaseUncased()
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    model = nn.DataParallel(model)

    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        outputs, labels = engine.eval_fn(test_data_loader, model, device)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(labels, outputs)
        print(f"Accuracy Score = {accuracy}")
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy


if __name__ == "__main__":
    run()
