import torch
import wandb
import torch.nn as nn

class Trainer():
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        criterion,
        device='cuda'
    ):
        model.to(device)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device


    def acc(self, outputs, labels):
        preds = outputs.argmax(dim=1)
        correct = (preds == labels).float()
        if correct.ndim > 1:
            sample_acc = correct.mean(dim=tuple(range(1, correct.ndim)))
        else:
            sample_acc = correct
        batch_acc = sample_acc.sum()
        # batch_acc = correct.sum()
        return batch_acc.item()


    def train(self, num_epochs, train_loader, val_loader, save_path='./models/best_model.pth'):
        global_step = 0
        best_acc = 0
        for epoch in range(num_epochs):
            train_acc = 0
            self.model.train()
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_acc += self.acc(outputs, labels)

                wandb.log({
                    'train/loss': loss.item(),
                    'lr': self.optimizer.param_groups[0]['lr']
                }, step=global_step)
                global_step += 1
            
            self.scheduler.step()

            val_acc,val_loss = 0, 0
            self.model.eval()
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    outputs = self.model(imgs)
                    loss = self.criterion(outputs, labels)

                    val_acc += self.acc(outputs, labels)
                    val_loss += loss.item()*len(labels)

            train_acc /= len(train_loader.dataset)
            val_acc /= len(val_loader.dataset)
            val_loss /= len(val_loader.dataset)

            if val_acc > best_acc:
                best_acc = val_acc
                self.save_model(save_path)

            wandb.log({
                'train/accuracy': train_acc,
                'val/accuracy': val_acc,
                'val/loss': val_loss
                }, step=global_step)

            # print(f'Epoch {epoch+1}/{num_epochs}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}')
            
    def test(self, test_loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                preds = outputs.argmax(dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        return torch.cat(all_labels), torch.cat(all_preds)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
