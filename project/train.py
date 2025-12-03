import torch
import torch.optim as optim
import torch.nn.functional as F
from model import Model

def permute_dims(z):
    B, d = z.shape
    z_perm = z.clone()
    for j in range(d):
        perm = torch.randperm(B)
        z_perm[:, j] = z[perm, j]
    return z_perm

def train_model(dataloader, device, config):
    model_cfg = config.model
    train_cfg = config.train
    data_cfg = config.data

    model = Model(model_cfg).to(device)
    
    vae_optimizer = optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=train_cfg.get("vae_lr", 1e-4),
        betas=(0.9, 0.999)
    )
    
    disc_optimizer = optim.Adam(
        model.discriminator.parameters(),
        lr=train_cfg.get("disc_lr", 1e-4),
        betas=(0.5, 0.9)
    )
    
    gamma = model_cfg.get("gamma", 6.0)
    epochs = train_cfg["num_epochs"]
    losses = []
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        avg_loss = train_epoch(model, dataloader, vae_optimizer, disc_optimizer, device, gamma)
        losses.append(avg_loss)
        print(f"Average Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), "result.pt")
    print("Model saved to result.pt")

    return {
        "model_name": model.__class__.__name__,
        "num_params": sum(p.numel() for p in model.parameters()),
        "epochs": epochs,
        "final_loss": losses[-1],
        "best_loss": min(losses),
        "best_epoch": int(losses.index(min(losses)) + 1),
        "device": device,
        "result_file": "result.pt",
        "config_used": {
            "model": model_cfg,
            "train": train_cfg,
            "data": data_cfg
        }
    }

def train_epoch(model, dataloader, vae_optimizer, disc_optimizer, device, gamma):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, (features, labels) in enumerate(dataloader):
        features = features.to(device)
        labels = labels.to(device)
        
        x_flat = features[:, :, 0]
        
        mu, logvar = model.encoder(x_flat)
        z = model.reparameterize(mu, logvar)
        recon = model.decoder(z)
        
        vae_loss, recon_loss, kl_loss = model.compute_vae_loss(features, recon, mu, logvar)
        
        disc_logits = model.discriminator(z)
        tc_loss = (disc_logits[:, 0] - disc_logits[:, 1]).mean()
        
        total_vae_loss = vae_loss - gamma * tc_loss
        
        vae_optimizer.zero_grad()
        total_vae_loss.backward(retain_graph=True)
        vae_optimizer.step()
        
        z_perm = permute_dims(z.detach())
        
        disc_logits_real = model.discriminator(z.detach())
        disc_logits_fake = model.discriminator(z_perm)
        
        disc_loss = 0.5 * (
            F.cross_entropy(disc_logits_real, torch.zeros(z.size(0), dtype=torch.long, device=device)) +
            F.cross_entropy(disc_logits_fake, torch.ones(z_perm.size(0), dtype=torch.long, device=device))
        )
        
        disc_optimizer.zero_grad()
        disc_loss.backward()
        disc_optimizer.step()
        
        total_loss += vae_loss.item()
        num_batches += 1

    return total_loss / num_batches
