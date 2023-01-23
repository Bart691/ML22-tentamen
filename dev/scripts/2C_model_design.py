from datetime import datetime

import tensorboard as tb
import torch
from loguru import logger

from tentamen.data import datasets
from tentamen.model import Accuracy
from tentamen.settings import presets
from tentamen.train import trainloop


if __name__ == "__main__":
    logger.add(presets.logdir / "01.log")

    trainstreamer, teststreamer = datasets.get_arabic(presets)

    from tentamen.model import gru_attention_model
    from tentamen.settings import gruatt_modelConfig

    configs_gruatt_model = [
        gruatt_modelConfig(
            input=13,
            output=20,
            tunedir=presets.logdir,
            hidden=256,
            dropout=0.2,
            num_layers=4,
        ),
    ]

    for config in configs_gruatt_model:
        model = gru_attention_model(config.dict())  # type: ignore

        trainedmodel = trainloop(
            epochs=25,
            model=model,  # type: ignore
            optimizer=torch.optim.Adam,
            learning_rate=1e-3,
            loss_fn=torch.nn.CrossEntropyLoss(),
            metrics=[Accuracy()],
            train_dataloader=trainstreamer.stream(),
            test_dataloader=teststreamer.stream(),
            log_dir=presets.logdir,
            train_steps=len(trainstreamer),
            eval_steps=len(teststreamer),
        )

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = presets.modeldir / (timestamp + presets.modelname)
        logger.info(f"save model to {path}")
        torch.save(trainedmodel, path)
