import torch.nn.functional as F
import torch
import copy

def train_epoch(model, optimizer, data):
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index) 

    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask]) # loss only on labeled
    loss.backward()

    optimizer.step()

    return loss.item()


@torch.no_grad()
def eval_epoch(model, data):
    model.eval()

    out = model(data.x, data.edge_index) 

    pred = out.argmax(dim=1)

    loss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask]) 

    accuracies = {}
    for split, mask in [
        ("train", data.train_mask), 
        ("val", data.val_mask), 
        ("test", data.test_mask)]:
        
        correct = (pred[mask] == data.y[mask]).sum().item()
        acc = correct / mask.sum().item()
        accuracies[split] = acc
    
    return accuracies, loss.item()


def train_model(model_class, 
                data, 
                num_features,
                hidden_channels, 
                num_classes, 
                device, 
                epochs,
                early_stopping,
                patience,
                optimizer_name,
                lr,
                weight_decay,
                single_layer=False,
                verbose=False):
    
    model = model_class(
        in_channels=num_features,
        hidden_channels=hidden_channels,
        out_channels=num_classes,
        single_layer=single_layer
    ).to(device)

    model_name = model._get_name()
    if verbose: print(model)

    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay) 
    else:
        raise ValueError(f"Optimizer not suppoted: {optimizer_name}!")

    best_val_loss = float("inf")
    best_val_acc = -1.0
    best_state = None
    best_epoch = None
    epochs_no_improvement = 0

    history = {
        "train_loss_history": [],
        "val_loss_history": [],
        "train_acc_history": [],
        "val_acc_history": [],
        "test_acc_history": []
    }

    for epoch in range(1, epochs+1):
        train_loss = train_epoch(model, optimizer, data)
        accs, val_loss = eval_epoch(model, data)

        history["train_loss_history"].append(train_loss)
        history["val_loss_history"].append(val_loss)
        history["train_acc_history"].append(accs['train'])
        history["val_acc_history"].append(accs['val'])
        history["test_acc_history"].append(accs['test'])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = accs['val']
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improvement = 0
        else:
            epochs_no_improvement += 1

        if verbose and epoch % 10 == 0:
            print(
                f"Epoch [{epoch:03d}/{epochs:03d}] | "
                f"Train loss: {train_loss:>8.4f} | "
                f"Val loss: {val_loss:>8.4f} | "
                f"Train accuracy: {accs['train']:>8.4f} | "
                f"Val accuracy: {accs['val']:>8.4f} | "
                f"Test accuracy: {accs['test']:>8.4f} | "
            )
        
        if early_stopping and epochs_no_improvement > patience:
            if verbose: print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    model.load_state_dict(best_state)
    best_metrics, final_val_loss = eval_epoch(model, data) 

    return {
        "model": model,
        "best_state_dict": best_state,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "history": history,
    }