from pathlib import Path

from sklearn.metrics import f1_score, precision_score, recall_score

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms

import gym

from Atari_Representations_Benchmark.dqn_replay_dataset import MultiDQNReplayDataset

try:
    import wandb
except:
    pass


class ActionClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, n_classes: int):
        super(ActionClassifier, self).__init__()
        self.encoder = encoder
        self.n_classes = n_classes

        self.classifier = nn.LazyLinear(n_classes)
        self.classifier.weight.data.normal_(mean=0.0, std=0.01)
        self.classifier.bias.data.zero_()

    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x)


def evaluate(encoder,
    dataset_path,
    n_epochs = 100,
    experiment_name = "",
    batch_size = 512,
    n_workers = 8,
    gpu = None,
    skip_train = False,
    no_freeze = False,
    lr = 0.001,
    game = "Pong",
    dqn_checkpoints = [50],
    dqn_frames = 3,
    dqn_single_dataset_max_size = 100000,
    dqn_test_single_dataset_max_size = 10000,
):
    if wandb.run is not None:
        wandb.init(name = experiment_name)

    env = gym.make(f"{game}NoFrameskip-v4")
    dqn_n_actions = env.action_space.n
    env.close()

    for p in encoder.parameters():
        p.requires_grad = no_freeze

    model = ActionClassifier(encoder, dqn_n_actions)

    if gpu is not None:
        print("Using GPU: {} for training".format(gpu))
        torch.cuda.set_device(gpu)
        model.cuda(gpu)

    float_transform = transforms.ConvertImageDtype(torch.float)

    train_dataset = MultiDQNReplayDataset(data_path=Path(dataset_path),
                                    games = game,
                                    checkpoints = dqn_checkpoints,
                                    frames = dqn_frames,
                                    max_size = dqn_single_dataset_max_size,
                                    transform = float_transform)

    test_dataset = MultiDQNReplayDataset(Path(dataset_path),
                                    games = game,
                                    checkpoints = dqn_checkpoints,
                                    frames = dqn_frames,
                                    max_size = dqn_single_dataset_max_size,
                                    transform = float_transform,
                                    start_index=dqn_single_dataset_max_size + dqn_test_single_dataset_max_size + 10000)

    train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_workers,
    )
    test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=n_workers,
    )

    if not skip_train:
        train(model, train_loader, gpu, n_epochs, lr, dqn_n_actions)

    test(model, test_loader, gpu)


def train(model, loader, gpu, n_epochs, lr, dqn_n_actions):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr, # linear scaling rule
        momentum=0.9,
    )
    loss_fun = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        epoch_len = len(loader)

        # Train
        model.train()
        for i, (x, y) in enumerate(loader):
            optimizer.zero_grad()

            if gpu is not None:
                x = x.cuda(gpu, non_blocking=True)

            label = F.one_hot(y.type(torch.int64), dqn_n_actions).type(torch.float).cuda(gpu)
            pred = model(x)
            loss = loss_fun(pred, label)

            if i % 50 == 0:
                print(f"epoch: {epoch} - [{i}/{epoch_len}] - loss: {loss.item()}")
                wandb.log({"loss": loss.item()})

            loss.backward()
            optimizer.step()


def test(model, loader, gpu):
    model.eval()
    y_pred = []
    y_true = []

    for i, (x, y) in enumerate(loader):
        if gpu is not None:
            x = x.cuda(gpu, non_blocking=True)
        y_true.extend(list(y.cpu().numpy()))
        with torch.no_grad():
            pred = torch.argmax(model(x), dim=1).to('cpu')
            y_pred.extend(list(pred.numpy()))

    total = len(y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    print(f"Total: {total}, Precision: {precision}, Recall: {recall}, F1 {f1}")
    wandb.log({"total": total, "precision": precision, "recall": recall, "f1": f1})
