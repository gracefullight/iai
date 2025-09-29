import os
import sys
from pathlib import Path

import cv2 as cv
import torch
from heading import *  # noqa: F403 (legacy import style kept)
from torchvision import datasets

# Allow overriding dataset root via env var (Path version)
_here = Path(__file__).parent
DATA_ROOT = Path(os.environ.get("FASHION_MNIST_DATA", str(_here / "Data")))
DATA_ROOT.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 10


def ask_model_option():
    env_val = os.environ.get("MODEL_OPTION")
    if env_val is not None:
        try:
            return int(env_val)
        except ValueError:
            print("Ignoring invalid MODEL_OPTION env var; falling back to input.")
    return int(
        input("Choose a model: 1 self-defined CNN; 2 AlexNet; 3 ResNet; 4 VGGNet.\nYour choice: ")
    )


model_option = ask_model_option()

# Robust image size selection with validation
_image_size_map = {1: 28, 3: 28, 2: 227, 4: 224}
image_size = _image_size_map.get(model_option)
if image_size is None:
    print(f"invalid model option: {model_option}")
    sys.exit(1)

transform = resize_image(image_size)


def load_data(transform, BATCH_SIZE, train, PATH=DATA_ROOT, download=True):
    """Return a DataLoader for FashionMNIST.

    Ensures the dataset is downloaded to PATH (default: DATA_ROOT).
    """
    load_dataset = datasets.FashionMNIST(
        root=PATH, train=train, transform=transform, download=download
    )

    data_loader = torch.utils.data.DataLoader(
        dataset=load_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    return data_loader


def build_loaders():
    train_loader = load_data(transform, BATCH_SIZE, train=True)
    test_loader = load_data(transform, BATCH_SIZE, train=False)
    return train_loader, test_loader


train_loader, test_loader = build_loaders()

label_list = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


image_dir = _here / "Image"

dataSet = FlameSet(str(image_dir), image_size)
select_image = dataSet[3]
img_path = image_dir / "6-0.png"
img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
if img is None or img.size == 0:
    print(f"Failed to read image: {img_path}")
else:
    cv.imshow("image", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


# Only be used with model_option = 1
class CONV_net(torch.nn.Module):
    def __init__(self, input_channels=1, output_size=10):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 16, 3, 2, 1),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 3, 1, 1),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
        )
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(288, 60), torch.nn.Dropout(0.5), torch.nn.ReLU()
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(60, 20), torch.nn.Dropout(0.5), torch.nn.ReLU()
        )
        # Explicit dim argument for Softmax to avoid deprecation warning
        self.fc3 = torch.nn.Sequential(torch.nn.Linear(20, output_size), torch.nn.Softmax(dim=1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 288)  # -1 tells PyTorch to automatically calculate the batch size dimension
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


input_channels = 1
output_size = 10
number_of_images = 100

if model_option == 1:  # self defined CNN
    convolution = CONV_net(input_channels, output_size)
    net_work = create_network(convolution)  # build a convolutional neural network

    # Set training parameters
    EPOCH = 1
    LR = 0.001

    # train the model
    trained_model = train_model(net_work, train_loader, LR, EPOCH)

    """
    Test the model using the test dataset loaded by using the test_loader.
    Without specifying the number of images used, you use all the images from the test dataset.

    """
    # test_model(trained_model, test_loader)
    test_model(
        trained_model,
        test_loader,
        number_of_images=10,
        label_names=label_list,
        print_confusions=True,
    )
    # classify your selected image sample
    predict_image(trained_model, select_image, label_list)
elif model_option == 2:  # AlexNet
    Alex_net = AlexNet(input_channels, output_size)
    alex_net = create_network(Alex_net)

    # Prepare parameters for treining
    EPOCH = 1
    LR = 0.001

    """
    Three modes of developing an AlexNet model:
    mode=1 train the model from scratch;
    mode=2, use a pretrained model and
    mode = 3 fine tunning a pretrained model
    """
    mode = int(
        input(
            "Choose a mode for model training: 1 for training from scratch, 2 for pretrained and 3 for fine-tuning based on a pretrained model, \n your choice is: "
        )
    )
    if mode == 1:
        trained_alexnet_model = train_model(alex_net, train_loader, LR, EPOCH, number_of_images)
        test_model(
            trained_alexnet_model,
            test_loader,
            number_of_images=10,
            label_names=label_list,
            print_confusions=True,
        )
        # test_model(trained_alexnet_model, test_loader, label_list, compute_auc=False)
        predict_image(trained_alexnet_model, select_image, label_list)
        # Save the trained model so that mode 2 can load it later
        try:
            torch.save(trained_alexnet_model, "AlexNet_pretrained_model.pth")
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] Could not save AlexNet model: {e}")
    elif mode == 2:
        pretrained_path = Path("AlexNet_pretrained_model.pth")
        if not pretrained_path.is_file():
            print(
                f"[WARN] Pretrained AlexNet weights not found at {pretrained_path.resolve()}. "
                "Run mode 1 first or supply the file. Skipping inference."
            )
        else:
            pretrained_alexnet_model = torch.load(
                str(pretrained_path), map_location=torch.device("cpu"), weights_only=False
            )
            test_model(
                pretrained_alexnet_model,
                test_loader,
                number_of_images=10,
                label_names=label_list,
                print_confusions=True,
            )
            predict_image(pretrained_alexnet_model, select_image, label_list)
    elif mode == 3:
        pretrained_alexnet_model = torch.load(
            "AlexNet_pretrained_model.pth", map_location=torch.device("cpu"), weights_only=False
        )
        re_trained_alexnet_model = train_model(
            pretrained_alexnet_model, train_loader, LR, EPOCH, number_of_images
        )
        # test the model using the test dataset loaded by using the test_loader
        test_model(
            re_trained_alexnet_model,
            test_loader,
            number_of_images=10,
            label_names=label_list,
            print_confusions=True,
        )
        # test_model(re_trained_alexnet_model , test_loader, label_list, compute_auc=False)
        # classify your selected image samples
        predict_image(re_trained_alexnet_model, select_image, label_list)
    else:
        print("invalid training mode")
elif model_option == 3:  # ResNet
    ResNet18 = ResNet(BasicBlock, [2, 2, 2, 2], input_channels, output_size)

    # create one version of ResNet18
    res_net_18 = create_network(ResNet18)

    # Prepare parameters for treining
    EPOCH = 1
    LR = 0.001

    """
    Three modes of developing a ResNet model:
    mode=1 train the model from scratch;
    mode=2, use a pretrained model and
    mode=4 fine tunning a pretrained model
    """
    mode = int(
        input(
            "Choose a mode for model training: 1 for training from scratch, 2 for pretrained and 3 for fine-tuning based on a pretrained model, \n your choice is: "
        )
    )
    if mode == 1:
        trained_model_resnet18 = train_model(res_net_18, train_loader, LR, EPOCH, number_of_images)
        # test the model using the test dataset loaded by using the test_loader
        test_model(
            trained_model_resnet18,
            test_loader,
            number_of_images=10,
            label_names=label_list,
            print_confusions=True,
        )
        # test_model(trained_model_resnet18, test_loader, label_list, compute_auc=False)
        # classify your selected image samples
        predict_image(trained_model_resnet18, select_image, label_list)
        try:
            torch.save(trained_model_resnet18, "ResNet18_pretrained_model.pth")
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] Could not save ResNet18 model: {e}")
    elif mode == 2:
        pretrained_path = Path("ResNet18_pretrained_model.pth")
        if not pretrained_path.is_file():
            print(
                f"[WARN] Pretrained ResNet18 weights not found at {pretrained_path.resolve()}. "
                "Run mode 1 first or supply the file. Skipping inference."
            )
        else:
            pretrained_model_resnet18 = torch.load(
                str(pretrained_path), map_location=torch.device("cpu"), weights_only=False
            )
            test_model(
                pretrained_model_resnet18,
                test_loader,
                number_of_images=10,
                label_names=label_list,
                print_confusions=True,
            )
            predict_image(pretrained_model_resnet18, select_image, label_list)
    elif mode == 3:
        pretrained_path = Path("ResNet18_pretrained_model.pth")
        if not pretrained_path.is_file():
            print(
                f"[WARN] Fine-tuning skipped: pretrained ResNet18 not found at {pretrained_path.resolve()}\n"
                "Run mode 1 first to create it (or place the file)."
            )
        else:
            pretrained_model_resnet18 = torch.load(
                str(pretrained_path), map_location=torch.device("cpu"), weights_only=False
            )
            re_trained_model_resnet18 = train_model(
                pretrained_model_resnet18, train_loader, LR, EPOCH, number_of_images
            )
            test_model(
                re_trained_model_resnet18,
                test_loader,
                number_of_images=10,
                label_names=label_list,
                print_confusions=True,
            )
            predict_image(re_trained_model_resnet18, select_image, label_list)
    else:
        print("invalid training mode")
elif model_option == 4:  # Pretrained VGGNet
    # Define the network configurations, cfg, where 'A' for VGG11, 'B' for VGG13, 'D' for VGG16 and 'E' for VGG19.
    cfg = {
        "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        "D": [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
            "M",
        ],
        "E": [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
            512,
            "M",
        ],
    }

    VGGNet11 = VGG(make_layers(cfg["A"], input_channels), num_classes=output_size)
    vgg_net_11 = create_network(VGGNet11)

    # Prepare parameters for treining
    EPOCH = 1
    LR = 0.001

    """
    Three modes of developing a VGGNet model:
    mode=1 train the model from scratch;
    mode=2, use a pretrained model and
    mode = 3 fine tunning a pretrained model
    """
    mode = int(
        input(
            "Choose a mode for model training: 1 for training from scratch, 2 for pretrained and 3 for fine-tuning based on a pretrained model, \n your choice is: "
        )
    )
    if mode == 1:
        trained_vgg_model_11 = train_model(vgg_net_11, train_loader, LR, EPOCH, number_of_images)

        # test the model using the test dataset loaded by using the test_loader
        test_model(
            trained_vgg_model_11,
            test_loader,
            number_of_images=10,
            label_names=label_list,
            print_confusions=True,
        )
        # test_model(trained_vgg_model_11, test_loader, label_list, compute_auc=False)
        torch.save(trained_vgg_model_11, "VGGNet11_model.pth")

        saved_vgg_model_11 = torch.load(
            "VGGNet11_model.pth", map_location=torch.device("cpu"), weights_only=False
        )

        # classify your selected image samples
        predict_image(saved_vgg_model_11, select_image, label_list)

    elif mode == 2:
        pretrained_path = Path("VGGNet11_pretrained_model.pth")
        if not pretrained_path.is_file():
            print(
                f"[WARN] Pretrained VGGNet11 weights not found at {pretrained_path.resolve()}. "
                "Run mode 1 first (saves a model) or provide the file. Skipping inference."
            )
        else:
            pretrained_vgg_model_11 = torch.load(
                str(pretrained_path), map_location=torch.device("cpu"), weights_only=False
            )
            test_model(
                pretrained_vgg_model_11,
                test_loader,
                number_of_images=10,
                label_names=label_list,
                print_confusions=True,
            )
            predict_image(pretrained_vgg_model_11, select_image, label_list)

    elif mode == 3:
        pretrained_path = Path("VGGNet11_pretrained_model.pth")
        if not pretrained_path.is_file():
            print(
                f"[WARN] Fine-tuning skipped: pretrained VGGNet11 not found at {pretrained_path.resolve()}\n"
                "Run mode 1 first (saves a model) or provide the pretrained file."
            )
        else:
            pretrained_vgg_model_11 = torch.load(
                str(pretrained_path), map_location=torch.device("cpu"), weights_only=False
            )
            re_trained_vgg_model_11 = train_model(
                pretrained_vgg_model_11, train_loader, LR, EPOCH, number_of_images
            )
            test_model(
                re_trained_vgg_model_11,
                test_loader,
                number_of_images=10,
                label_names=label_list,
                print_confusions=True,
            )
            predict_image(re_trained_vgg_model_11, select_image, label_list)
    else:
        print("invalid training mode")


if __name__ == "__main__":
    print(f"Model option: {model_option}; image size: {image_size}; data root: {DATA_ROOT}")
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
