from pathlib import Path

import fire
import torch
from sklearn.metrics import f1_score, roc_auc_score

from mti_anomaly_detection.datahandler import get_dataloader_single_folder
from mti_anomaly_detection.model import createDeepLabv3
from mti_anomaly_detection.trainer import train_model


class AnomalyDetectionApp:
    def train(
        self,
        data_directory: str,  # Specify the data directory
        exp_directory: str,  # Specify the experiment directory.
        epochs: int = 25,  # Specify the number of epochs you want to run the experiment for
        batch_size: int = 4,  # Specify the batch size for the dataloader
    ):
        model = createDeepLabv3()
        model.train()
        data_directory = Path(data_directory)
        exp_directory = Path(exp_directory)
        if not exp_directory.exists():
            exp_directory.mkdir()

        criterion = torch.nn.MSELoss(reduction="mean")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        metrics = {"f1_score": f1_score, "auroc": roc_auc_score}

        dataloaders = get_dataloader_single_folder(
            data_directory, batch_size=batch_size
        )
        _ = train_model(
            model,
            criterion,
            dataloaders,
            optimizer,
            bpath=exp_directory,
            metrics=metrics,
            num_epochs=epochs,
        )

        torch.save(model, exp_directory / "weights.pt")


if __name__ == "__main__":
    fire.Fire(AnomalyDetectionApp)
