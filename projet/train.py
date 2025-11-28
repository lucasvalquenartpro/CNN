import torch
from torch import nn
from torch.utils.data import Subset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import time

import data
import model
import utils


class Config:
    """Inititalisation des variables importantes"""
    train_file = '../train.txt'
    train_dir = '../train/'
    save_dir = 'checkpoints'

    batch_size = 128
    num_epochs = 30
    learning_rate = 0.001

    patience = 5
    min_delta = 0.001

    scheduler_patience = 3
    scheduler_factor = 0.5

    seed = 42


def set_seed(seed):
    """Fixe la seed pour la reproductibilité"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def train_one_epoch(model, train_loader, criterion, optimizer, device, fairness_weight=0.5):
    """Entraîne le modèle sur une epoch"""
    model.train()

    total_loss = 0
    num_batches = 0

    for (images, genres), labels in train_loader:
        images = images.to(device)
        genres = genres.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions = model(images, genres)
        weights = torch.where(labels == 1, 1.5, 1.0)
        loss_unreduced = nn.functional.smooth_l1_loss(predictions, labels, reduction='none')
        main_loss = (loss_unreduced * weights).mean()

        # Pénalité de fairness : égaliser les erreurs entre genres
        male_mask = (genres == 1)
        female_mask = (genres == -1)

        if male_mask.sum() > 0 and female_mask.sum() > 0:
            male_loss = loss_unreduced[male_mask].mean()
            female_loss = loss_unreduced[female_mask].mean()
            fairness_penalty = torch.abs(male_loss - female_loss)
            loss = main_loss + fairness_weight * fairness_penalty
        else:
            # Si un seul genre dans le batch, utiliser uniquement la loss principale
            loss = main_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, val_loader, criterion, device):
    """Calcul les metrics du modèle sur une epoch"""
    model.eval()

    total_loss = 0
    num_batches = 0

    metrics_tracker = utils.Metrics()

    # Pour chaque batchs :
    with torch.no_grad():
        for (images, genres), labels in val_loader:
            images = images.to(device)
            genres = genres.to(device)
            labels = labels.to(device)

            predictions = model(images, genres)
            loss = criterion(predictions, labels)

            total_loss += loss.item()
            num_batches += 1

            metrics_tracker.update(predictions, labels, genres)

    # Metrics pour une epoch
    avg_loss = total_loss / num_batches
    metrics = metrics_tracker.compute()

    return avg_loss, metrics

if __name__ == '__main__':
    config = Config()
    set_seed(config.seed)

    # Config GPU/CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"device : {device}")

    # Enregistrement des données pour analyse tensorboard
    Path(config.save_dir).mkdir(exist_ok=True)
    writer = SummaryWriter('runs/face_classifier')

    # Config data
    df = data.dataframe(config.train_file)
    train_idx, val_idx = data.split_train_val_data(df)
    train_full_dataset = data.FaceDataset(
        df['img_paths'],
        df['genre'],
        df['label'],
        transform=data.transform(istrain=True)
    )

    val_full_dataset = data.FaceDataset(
        df['img_paths'],
        df['genre'],
        df['label'],
        transform=data.transform(istrain=False)
    )

    train_dataset = Subset(train_full_dataset, train_idx)
    val_dataset = Subset(val_full_dataset, val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    # Config Modele et hyperparamètre
    net = model.ResNet(pretrained=True, freeze_early_layers=True).to(device)
    criterion = utils.get_loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=config.scheduler_patience,
        factor=config.scheduler_factor,
        verbose=True
    )

    print(f"""
        Configuration:
        Optimizer: Adam (LR={config.learning_rate})
        Scheduler: ReduceLROnPlateau (patience={config.scheduler_patience})
        Loss: SmoothL1Loss
        Early stopping patience: {config.patience}""")

    # Early stopping
    best_fairness_score = -float('inf')
    epochs_without_improvement = 0

    print("ENTRAÎNEMENT :")

    start_time = time.time()

    for epoch in range(config.num_epochs):
        epoch_start = time.time()

        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        print("-" * 60)

        # Entrainement
        train_loss = train_one_epoch(net, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate(net, val_loader, criterion, device)
        scheduler.step(val_metrics['fairness_score'])
        epoch_time = time.time() - epoch_start

        # Affichage terminal
        print(f"""
            Temps: {epoch_time:.1f}s
            Train Loss: {train_loss:.4f}
            Val Loss: {val_loss:.4f}
            Val Metrics:
            Acc Global: {val_metrics['acc_global']:.4f}
            Acc Hommes: {val_metrics['acc_male']:.4f}
            Acc Femmes: {val_metrics['acc_female']:.4f}
            Fairness Score: {val_metrics['fairness_score']:.4f}""")

        # Affichage TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metrics/acc_global', val_metrics['acc_global'], epoch)
        writer.add_scalar('Metrics/acc_male', val_metrics['acc_male'], epoch)
        writer.add_scalar('Metrics/acc_female', val_metrics['acc_female'], epoch)
        writer.add_scalar('Metrics/fairness_score', val_metrics['fairness_score'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Early stopping
        if val_metrics['fairness_score'] > best_fairness_score + config.min_delta:
            print(f"Nouveau meilleur score! ({best_fairness_score:.4f} → {val_metrics['fairness_score']:.4f})")
            best_fairness_score = val_metrics['fairness_score']
            epochs_without_improvement = 0

            # Sauvegarder le meilleur modèle
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'fairness_score': best_fairness_score,
                'metrics': val_metrics
            }, f'{config.save_dir}/best_model.pth')
        else:
            epochs_without_improvement += 1
            print(f"Pas d'amélioration ({epochs_without_improvement}/{config.patience})")

            if epochs_without_improvement >= config.patience:
                print(f"\nEarly stopping après {epoch + 1} epochs")
                break

    total_time = time.time() - start_time

    print(f"""
        ENTRAÎNEMENT TERMINÉ
        Temps total: {total_time / 60:.1f} minutes
        Meilleur Fairness Score: {best_fairness_score:.4f}""")

    writer.close()