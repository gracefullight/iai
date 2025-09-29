import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn, optim
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms

# ___________________________________________
# ResNet


# For ResNet 18, 34, use two 3 x 3 convolution
class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.shortcut = torch.nn.Sequential()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
                ),
                torch.nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = torch.nn.functional.relu(self.bn2(self.conv2(out)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.nn.functional.relu(out)
        return out


class ResNet(torch.nn.Module):
    def __init__(
        self, block: type, num_blocks: list[int], input_channels: int = 1, num_classes: int = 10
    ) -> None:
        super().__init__()
        self.in_planes = 64

        self.conv1 = torch.nn.Conv2d(
            input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = torch.nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self, block: type, planes: int, num_blocks: int, stride: int
    ) -> torch.nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.nn.functional.avg_pool2d(out, 4)

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# ___________________________________________
# VGG 11


class VGG(nn.Module):
    def __init__(
        self, features: nn.Sequential, num_classes: int = 10, init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(
    cfg: list[int | str], input_channels: int = 1, batch_norm: bool = False
) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_channels: int = input_channels
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, int(v), kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(int(v)), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = int(v)
    return nn.Sequential(*layers)


class AlexNet(torch.nn.Module):
    def __init__(self, input_channels: int = 1, output_size: int = 10) -> None:
        super().__init__()
        self.conv1 = torch.nn.Sequential(  # input_size = 227*227*1
            torch.nn.Conv2d(input_channels, 96, 11, 4, 0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),  # output_size = 27*27*96
        )
        self.conv2 = torch.nn.Sequential(  # input_size = 27*27*96
            torch.nn.Conv2d(96, 256, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2),  # output_size = 13*13*256
        )
        self.conv3 = torch.nn.Sequential(  # input_size = 13*13*256
            torch.nn.Conv2d(256, 384, 3, 1, 1),
            torch.nn.ReLU(),  # output_size = 13*13*384
        )
        self.conv4 = torch.nn.Sequential(  # input_size = 13*13*384
            torch.nn.Conv2d(384, 384, 3, 1, 1),
            torch.nn.ReLU(),  # output_size = 13*13*384
        )
        self.conv5 = torch.nn.Sequential(  # input_size = 13*13*384
            torch.nn.Conv2d(384, 256, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2),  # output_size = 6*6*256
        )

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(9216, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        conv5_out = self.conv5(x)
        x = conv5_out.view(conv5_out.size(0), -1)
        out = self.dense(x)
        return out


class FlameSet(data.Dataset):
    global image_size

    def __init__(self, root: str, image_size: int) -> None:
        self.imgs_list = os.listdir(root)
        self.imgs = [os.path.join(root, k) for k in self.imgs_list]
        self.labels = [k.split("-")[0] for k in self.imgs_list]
        self.transforms = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])

    def __getitem__(self, index: int) -> torch.Tensor:
        img_path = self.imgs[index]
        pil_img = Image.open(img_path)
        pil_img = pil_img.convert("L")
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img)
            data = torch.from_numpy(pil_img)
        return data

    def __len__(self) -> int:
        return len(self.imgs)

    def get_image_list(self) -> list[str]:
        return self.imgs_list

    def get_image_label(self) -> list[str]:
        return self.labels


def resize_image(input_size: int = 28) -> transforms.Compose:
    transform = transforms.Compose([transforms.Resize(input_size), transforms.ToTensor()])
    return transform


def view_datasets(
    image_loader: torch.utils.data.DataLoader, objective_list: list[str]
) -> tuple[torch.Tensor, np.ndarray]:
    objective_list = np.array(objective_list)
    images, labels = next(iter(image_loader))
    img = torchvision.utils.make_grid(images)
    img = img.numpy().transpose(1, 2, 0)
    print(objective_list[labels.tolist()])
    # plt.axis('off')
    # plt.imshow(img)
    return (images, objective_list[labels.tolist()])


def create_network(network: nn.Module) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = network
    net = res.to(device)
    return net


def train_model(
    net: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    LR: float,
    epochs: int = 1,
    number_of_images: int | None = None,
) -> nn.Module:
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    for epoch in range(epochs):
        sum_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            else:
                inputs, labels = Variable(inputs).cpu(), Variable(labels).cpu()
            optimizer.zero_grad()  # Make gradient to zero
            outputs = net(inputs)  # Forward calculation
            loss = loss_function(outputs, labels)  # Get loss function
            loss.backward()  # back propogation
            optimizer.step()  # Update parameter.
            # print(loss)
            sum_loss += loss.item()
            if i % 100 == 99:
                print("[%d,%d] loss:%.03f" % (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0
            if number_of_images is None:
                pass
            elif i * train_loader.batch_size >= number_of_images:
                print(f"Current Loss {sum_loss}")
                break
    return net


def test_model(
    net: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    number_of_images: int | None = None,
    label_names: list[str] | None = None,
    print_confusions: bool = False,
) -> dict[str, float]:
    """Prints a full evaluation report:
    - Accuracy, weighted Precision/Recall/F1
    - Top-3 and Top-5 accuracy
    - Classification report (per-class metrics)
    - (optional) Top confusions text
    """
    net.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    total: int = 0
    correct: int = 0
    top3_correct: int = 0
    top5_correct: int = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            else:
                images, labels = images.cpu(), labels.cpu()

            logits = net(images)
            _, predicted = torch.max(logits, 1)

            # Respect number_of_images cap
            if number_of_images is not None:
                remaining = number_of_images - total
                if remaining <= 0:
                    break
                if labels.size(0) > remaining:
                    labels = labels[:remaining]
                    predicted = predicted[:remaining]
                    logits = logits[:remaining]

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

            # Top-k correctness (per-batch)
            top3 = torch.topk(logits, k=3, dim=1).indices
            top5 = torch.topk(logits, k=5, dim=1).indices
            top3_correct += (top3 == labels.unsqueeze(1)).any(dim=1).sum().item()
            top5_correct += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()

            if number_of_images is not None and total >= number_of_images:
                break

    # Overall metrics
    acc: float = accuracy_score(all_labels, all_preds) if total else 0.0
    prec: float = (
        precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        if total
        else 0.0
    )
    rec: float = (
        recall_score(all_labels, all_preds, average="weighted", zero_division=0) if total else 0.0
    )
    f1: float = (
        f1_score(all_labels, all_preds, average="weighted", zero_division=0) if total else 0.0
    )
    top3_acc: float = (top3_correct / total) if total else 0.0
    top5_acc: float = (top5_correct / total) if total else 0.0
    print(f"Correct predictions: {correct}")
    print(f"Total samples:      {total}")
    print(f"Accuracy:           {acc * 100:.2f}%")
    print(f"Precision (weighted): {prec:.4f}")
    print("\n--- Classification Report ---")
    try:
        if label_names is not None:
            print(
                classification_report(
                    all_labels, all_preds, target_names=label_names, zero_division=0
                )
            )
        else:
            print(classification_report(all_labels, all_preds, zero_division=0))
    except ValueError:
        # Fallback if label_names length/order doesn't match labels encountered
        print(classification_report(all_labels, all_preds, zero_division=0))

    # Optional: list top confusions (text)
    if print_confusions:
        cm = confusion_matrix(all_labels, all_preds)
        off_diag: list[tuple[int, int, int]] = []
        # Collect off-diagonal confusion counts (ensure consistent 4-space indentation)
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                if r != c and cm[r, c] > 0:
                    off_diag.append((r, c, cm[r, c]))
        off_diag.sort(key=lambda x: x[2], reverse=True)
        tops = off_diag
        if tops:
            print("\nTop confusions (true → predicted : count):")
            for r, c, cnt in tops:
                tr = label_names[r] if label_names and r < len(label_names) else str(r)
                pr = label_names[c] if label_names and c < len(label_names) else str(c)
                print(f"  {tr} → {pr} : {cnt}")

    # Return if you want to capture programmatically
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "top3": top3_acc,
        "top5": top5_acc,
    }


def predict_image(
    net: nn.Module,
    input_image: torch.Tensor | torch.utils.data.DataLoader,
    objective_list: list[str],
    num_of_prediction: int = 1,
    topk: int = 5,
) -> None:
    """input_image: either a single transformed tensor (C,H,W) or a DataLoader
    objective_list: list of class names (index-aligned with your labels)
    """
    net.eval()
    objective_list = np.array(objective_list)

    with torch.no_grad():
        if isinstance(input_image, torch.utils.data.DataLoader):
            # Show predictions for the first num_of_prediction samples from the loader
            shown: int = 0
            for images, _ in input_image:
                if torch.cuda.is_available():
                    images = images.cuda()
                    net = net.cuda()
                logits = net(images)
                probs = F.softmax(logits, dim=1)
                _, top1 = torch.max(probs, 1)
                for b in range(images.size(0)):
                    if shown >= num_of_prediction:
                        return
                    top_vals, top_idx = torch.topk(probs[b], k=min(topk, probs.size(1)))
                    top_vals = top_vals.cpu().numpy()
                    top_idx = top_idx.cpu().numpy()
                    top_labels = objective_list[top_idx]
                    print(f"# Sample {shown + 1}")
                    print(
                        f"Top-1: {objective_list[top1[b].item()]} ({probs[b, top1[b]].item() * 100:.1f}%)"
                    )
                    print(
                        "Top-{}: {}".format(
                            len(top_labels),
                            ", ".join(
                                [
                                    f"{lbl} ({val * 100:.1f}%)"
                                    for lbl, val in zip(top_labels, top_vals, strict=False)
                                ]
                            ),
                        )
                    )
                    shown += 1
                if shown >= num_of_prediction:
                    return
        else:
            # Single image tensor path
            if input_image.ndim == 3:
                # shape (C,H,W) expected; original code did unsqueeze(-3) which can be awkward
                image_batch = input_image.unsqueeze(0)  # (1,C,H,W)
            else:
                image_batch = input_image  # assume already (1,C,H,W)

            if torch.cuda.is_available():
                net = net.cuda()
                image_batch = image_batch.cuda()

            logits = net(image_batch)
            probs = F.softmax(logits, dim=1)
            top_vals, top_idx = torch.topk(probs[0], k=min(topk, probs.size(1)))
            top_vals = top_vals.cpu().numpy()
            top_idx = top_idx.cpu().numpy()
            top_labels = objective_list[top_idx]

            # Clean single-line prints
            print(f"Top-1 prediction: {top_labels[0]} ({top_vals[0] * 100:.1f}%)")
            print(
                "Top-{}: {}".format(
                    len(top_labels),
                    ", ".join(
                        [
                            f"{lbl} ({val * 100:.1f}%)"
                            for lbl, val in zip(top_labels, top_vals, strict=False)
                        ]
                    ),
                )
            )
