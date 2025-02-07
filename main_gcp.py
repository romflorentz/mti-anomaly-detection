from pathlib import Path
import fire
from sklearn.metrics import f1_score, roc_auc_score
import copy
import csv
import os
import time
import numpy as np
from tqdm import tqdm
import torch
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Any, Callable, Optional
from PIL import Image
from torchvision.datasets.vision import VisionDataset


class SegmentationDataset(VisionDataset):
    """A PyTorch dataset for image segmentation task.
    The dataset is compatible with torchvision transforms.
    The transforms passed would be applied to both the Images and Masks.
    """

    def __init__(
        self,
        root: str,
        image_folder: str,
        mask_folder: str,
        transforms: Optional[Callable] = None,
        seed: int = None,
        fraction: float = None,
        subset: str = None,
        image_color_mode: str = "rgb",
        mask_color_mode: str = "grayscale",
    ) -> None:
        """
        Args:
            root (str): Root directory path.
            image_folder (str): Name of the folder that contains the images in the root directory.
            mask_folder (str): Name of the folder that contains the masks in the root directory.
            transforms (Optional[Callable], optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.ToTensor`` for images. Defaults to None.
            seed (int, optional): Specify a seed for the train and test split for reproducible results. Defaults to None.
            fraction (float, optional): A float value from 0 to 1 which specifies the validation split fraction. Defaults to None.
            subset (str, optional): 'Train' or 'Test' to select the appropriate set. Defaults to None.
            image_color_mode (str, optional): 'rgb' or 'grayscale'. Defaults to 'rgb'.
            mask_color_mode (str, optional): 'rgb' or 'grayscale'. Defaults to 'grayscale'.

        Raises:
            OSError: If image folder doesn't exist in root.
            OSError: If mask folder doesn't exist in root.
            ValueError: If subset is not either 'Train' or 'Test'
            ValueError: If image_color_mode and mask_color_mode are either 'rgb' or 'grayscale'
        """
        super().__init__(root, transforms)
        image_folder_path = Path(self.root) / image_folder
        mask_folder_path = Path(self.root) / mask_folder
        if not image_folder_path.exists():
            raise OSError(f"{image_folder_path} does not exist.")
        if not mask_folder_path.exists():
            raise OSError(f"{mask_folder_path} does not exist.")

        if image_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(
                f"{image_color_mode} is an invalid choice. Please enter from rgb grayscale."
            )
        if mask_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(
                f"{mask_color_mode} is an invalid choice. Please enter from rgb grayscale."
            )

        self.image_color_mode = image_color_mode
        self.mask_color_mode = mask_color_mode

        if not fraction:
            self.image_names = sorted(image_folder_path.glob("*"))
            self.mask_names = sorted(mask_folder_path.glob("*"))
        else:
            if subset not in ["Train", "Test"]:
                raise (
                    ValueError(
                        f"{subset} is not a valid input. Acceptable values are Train and Test."
                    )
                )
            self.fraction = fraction
            self.image_list = np.array(sorted(image_folder_path.glob("*")))
            self.mask_list = np.array(sorted(mask_folder_path.glob("*")))
            if seed:
                np.random.seed(seed)
                indices = np.arange(len(self.image_list))
                np.random.shuffle(indices)
                self.image_list = self.image_list[indices]
                self.mask_list = self.mask_list[indices]
            if subset == "Train":
                self.image_names = self.image_list[
                    : int(np.ceil(len(self.image_list) * (1 - self.fraction)))
                ]
                self.mask_names = self.mask_list[
                    : int(np.ceil(len(self.mask_list) * (1 - self.fraction)))
                ]
            else:
                self.image_names = self.image_list[
                    int(np.ceil(len(self.image_list) * (1 - self.fraction))) :
                ]
                self.mask_names = self.mask_list[
                    int(np.ceil(len(self.mask_list) * (1 - self.fraction))) :
                ]

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> Any:
        image_path = self.image_names[index]
        mask_path = self.mask_names[index]
        with open(image_path, "rb") as image_file, open(mask_path, "rb") as mask_file:
            image = Image.open(image_file)
            if self.image_color_mode == "rgb":
                image = image.convert("RGB")
            elif self.image_color_mode == "grayscale":
                image = image.convert("L")
            mask = Image.open(mask_file)
            if self.mask_color_mode == "rgb":
                mask = mask.convert("RGB")
            elif self.mask_color_mode == "grayscale":
                mask = mask.convert("L")
            sample = {"image": image, "mask": mask}
            if self.transforms:
                sample["image"] = self.transforms(sample["image"])
                sample["mask"] = self.transforms(sample["mask"])
            return sample


def get_dataloader_sep_folder(
    data_dir: str,
    image_folder: str = "Image",
    mask_folder: str = "Mask",
    batch_size: int = 4,
):
    """Create Train and Test dataloaders from two
        separate Train and Test folders.
        The directory structure should be as follows.
        data_dir
        --Train
        ------Image
        ---------Image1
        ---------ImageN
        ------Mask
        ---------Mask1
        ---------MaskN
        --Test
        ------Image
        ---------Image1
        ---------ImageM
        ------Mask
        ---------Mask1
        ---------MaskM

    Args:
        data_dir (str): The data directory or root.
        image_folder (str, optional): Image folder name. Defaults to 'Image'.
        mask_folder (str, optional): Mask folder name. Defaults to 'Mask'.
        batch_size (int, optional): Batch size of the dataloader. Defaults to 4.

    Returns:
        dataloaders: Returns dataloaders dictionary containing the
        Train and Test dataloaders.
    """
    data_transforms = transforms.Compose([transforms.ToTensor()])

    image_datasets = {
        x: SegmentationDataset(
            root=Path(data_dir) / x,
            transforms=data_transforms,
            image_folder=image_folder,
            mask_folder=mask_folder,
        )
        for x in ["Train", "Test"]
    }
    dataloaders = {
        x: DataLoader(
            image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0
        )
        for x in ["Train", "Test"]
    }
    return dataloaders


def get_dataloader_single_folder(
    data_dir: str,
    image_folder: str = "Images",
    mask_folder: str = "Masks",
    fraction: float = 0.2,
    batch_size: int = 4,
):
    """Create train and test dataloader from a single directory containing
    the image and mask folders.

    Args:
        data_dir (str): Data directory path or root
        image_folder (str, optional): Image folder name. Defaults to 'Images'.
        mask_folder (str, optional): Mask folder name. Defaults to 'Masks'.
        fraction (float, optional): Fraction of Test set. Defaults to 0.2.
        batch_size (int, optional): Dataloader batch size. Defaults to 4.

    Returns:
        dataloaders: Returns dataloaders dictionary containing the
        Train and Test dataloaders.
    """
    data_transforms = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()]
    )

    image_datasets = {
        x: SegmentationDataset(
            data_dir,
            image_folder=image_folder,
            mask_folder=mask_folder,
            seed=100,
            fraction=fraction,
            subset=x,
            transforms=data_transforms,
        )
        for x in ["Train", "Test"]
    }
    dataloaders = {
        x: DataLoader(
            image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=8
        )
        for x in ["Train", "Test"]
    }
    return dataloaders


def createDeepLabv3(outputchannels=1):
    model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
    model.classifier = DeepLabHead(2048, outputchannels)
    model.train()
    return model


def train_model(model, criterion, dataloaders, optimizer, metrics, bpath, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    device = torch.device("cuda" if torch.mps.is_available() else "cpu")
    model.to(device)
    fieldnames = (
        ["epoch", "Train_loss", "Test_loss"]
        + [f"Train_{m}" for m in metrics.keys()]
        + [f"Test_{m}" for m in metrics.keys()]
    )
    with open(os.path.join(bpath, "log.csv"), "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(1, num_epochs + 1):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print("-" * 10)
        # Each epoch has a training and validation phase
        # Initialize batch summary
        batchsummary = {a: [0] for a in fieldnames}

        for phase in ["Train", "Test"]:
            if phase == "Train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # Iterate over data.
            for sample in tqdm(iter(dataloaders[phase])):
                inputs = sample["image"].to(device)
                masks = sample["mask"].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == "Train"):
                    outputs = model(inputs)
                    loss = criterion(outputs["out"], masks)
                    y_pred = outputs["out"].data.cpu().numpy().ravel()
                    y_true = masks.data.cpu().numpy().ravel()
                    for name, metric in metrics.items():
                        if name == "f1_score":
                            # Use a classification threshold of 0.1
                            batchsummary[f"{phase}_{name}"].append(
                                metric(y_true > 0, y_pred > 0.1)
                            )
                        else:
                            batchsummary[f"{phase}_{name}"].append(
                                metric(y_true.astype("uint8"), y_pred)
                            )

                    # backward + optimize only if in training phase
                    if phase == "Train":
                        loss.backward()
                        optimizer.step()
            batchsummary["epoch"] = epoch
            epoch_loss = loss
            batchsummary[f"{phase}_loss"] = epoch_loss.item()
            print("{} Loss: {:.4f}".format(phase, loss))
        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
        print(batchsummary)
        with open(os.path.join(bpath, "log.csv"), "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
            # deep copy the model
            if phase == "Test" and loss < best_loss:
                best_loss = loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Lowest Loss: {:4f}".format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


class AnomalyDetectionApp:
    def main(
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
