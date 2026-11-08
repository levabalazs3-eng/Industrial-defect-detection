import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm
from torchmetrics import Accuracy, ConfusionMatrix

class Trainer:
  def __init__(self, model, classes, loss_fn, lr, device):
    self.model = model
    self.classes = classes
    self.loss_fn = loss_fn
    self.device = device
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

  def fit(self, train_loader, val_loader=None, epochs=10, debug_plot=False):
    acc = Accuracy(task="multiclass", num_classes=len(self.classes))
    acc = acc.to(self.device)

    train_accs = []
    val_accs = []
    train_loss = []

    for epoch in range(epochs):
      acc.reset()
      pbar = tqdm(train_loader, desc="train")
      for img, lbl in pbar:
        img, lbl = img.to(self.device), lbl.to(self.device)
        if debug_plot:
          self._debug_plot(img, lbl, self.classes)
        probs, loss = self._train_step(img, lbl)
        batch_acc = acc(probs, lbl)
        train_loss.append(loss.item())
        pbar.set_postfix({"loss": loss.item(), "acc": batch_acc.item()})

      train_epoch_acc = acc.compute().detach()
      train_accs.append(train_epoch_acc)
      print(f"Epoch {epoch} train accuracy: {train_epoch_acc.item()}")
      acc.reset()

      if val_loader:
        pbar = tqdm(val_loader, desc="val")
        for img, lbl in pbar:
          img, lbl = img.to(self.device), lbl.to(self.device)
          if debug_plot:
            self._debug_plot(img, lbl, self.classes)
          probs, loss = self._val_step(img, lbl)
          batch_acc = acc(probs, lbl)
          pbar.set_postfix({"loss": loss.item(), "acc": batch_acc.item()})

        val_epoch_acc = acc.compute().detach()
        val_accs.append(val_epoch_acc)
        print(f"Epoch {epoch} val accuracy: {val_epoch_acc.item()}")

    return train_loss, train_accs, val_accs

  def validate(self, val_loader):

    acc = Accuracy(task="multiclass", num_classes=len(self.classes))
    acc = acc.to(self.device)

    confm = ConfusionMatrix(task="multiclass", num_classes=len(self.classes))
    confm = confm.to(self.device)

    pbar = tqdm(val_loader, desc="val")
    for img, lbl in pbar:
      img, lbl = img.to(self.device), lbl.to(self.device)
      probs, _ = self._val_step(img, lbl)
      acc(probs, lbl)
      confm(probs, lbl)
    return acc, confm


  def find_lr(self, train_loader, steps=100, log_range=(-5, 1), smoothing=5):

    log_values = torch.linspace(log_range[0], log_range[1], steps)
    lr_values = 10 ** log_values

    loader = iter(train_loader)
    losses = []

    for i in range(steps):
      try:
        img, lbl = next(loader)
      except StopIteration:
        data_iter = iter(train_loader)
        images, labels = next(data_iter)

      img, lbl = img.to(self.device), lbl.to(self.device)

      for param_group in self.optimizer.param_groups:
        param_group['lr'] = lr_values[i].item()

      _, loss = self._train_step(img, lbl)
      losses.append(loss.item())

    fig = plt.figure()
    ax = fig.gca()

    losses = np.array(losses)
    box = np.ones(smoothing) / smoothing
    losses = np.convolve(losses, box, mode='same')

    ax.set_xscale("log")
    ax.plot(lr_values.numpy(), losses)
    fig.show()

  def _train_step(self, img, lbl):

    res = self.model(img)
    loss = self.loss_fn(res, lbl)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return res, loss

  def _val_step(self, img, lbl):
    res = self.model(img)
    loss = self.loss_fn(res, lbl)

    return res, loss

def _debug_plot(self, img, lbl, classes):

    img_cpu = img.detach().to("cpu")
    grid = torchvision.utils.make_grid(img_cpu, nrow=8)
    fig = plt.figure()
    ax = fig.gca()
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.axis('off')
    plt.show()
    label_indices = lbl.detach().to("cpu").numpy().tolist()
    label_names = np.array(self.classes)[label_indices]
    print("Labels: ", label_names)

'''
Epoch 98 train accuracy: 0.8970588445663452
val: 100%|██████████████| 2/2 [00:12<00:00,  6.44s/it, loss=0.186, acc=0.909]
Epoch 98 val accuracy: 0.930232584476471
train: 100%|████████████████| 7/7 [00:15<00:00,  2.15s/it, loss=0.235, acc=1]
Epoch 99 train accuracy: 0.9166666865348816
val: 100%|█████████████████| 2/2 [00:12<00:00,  6.40s/it, loss=0.0776, acc=1]
Epoch 99 val accuracy: 0.9069767594337463
'''
