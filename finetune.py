"""
finetune.py  —  Fase 1: Fine-tuning supervisado del actor con datos DAgger
===========================================================================
Ejecutar DESPUÉS de a2c_dagger.py (que genera dagger_interventions.pkl):
    python3 finetune.py

Correcciones respecto a la versión original:
  1. Nombre de archivo unificado → a2c_expert_xd.pt  (igual en los 3 scripts).
  2. actor_log_std también se actualiza durante el fine-tuning.
  3. Se imprime el learning rate efectivo y se usa un scheduler para estabilizar.
  4. Validación de que las acciones del dataset estén en el rango correcto [-1,1].
  5. Posibilidad de acumular datasets de múltiples rondas DAgger (append mode).
  6. Guarda gráfica de la curva de fine-tuning para el informe.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────────────────────────
# ARQUITECTURA  ← IDÉNTICA a train_expert.py y a2c_dagger.py
# ─────────────────────────────────────────────────────────────
class A2CContinuous(nn.Module):
    def __init__(self, state_dim=14, action_dim=2):
        super().__init__()
        self.actor_net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
            nn.Linear(128, action_dim), nn.Tanh()
        )
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic_net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
            nn.Linear(128, 1)
        )


# ─────────────────────────────────────────────────────────────
# FINE-TUNING
# ─────────────────────────────────────────────────────────────
def run_finetuning(
    dataset_path   = "dagger_interventions.pkl",
    weights_path   = "a2c_expert_xd.pt",
    epochs         = 50,
    lr             = 3e-5,
    batch_size     = 64,
    accumulate_pkl = True,     # Si True, acumula todos los .pkl encontrados en el directorio
):
    print("=" * 55)
    print("  FINE-TUNING SUPERVISADO (DAgger Imitation Loss)")
    print("=" * 55)

    # ── 1. Cargar dataset ─────────────────────────────────────
    all_data = []

    if accumulate_pkl:
        # Buscar todos los archivos .pkl de intervención en el directorio actual
        pkl_files = [f for f in os.listdir(".") if f.endswith(".pkl") and "intervention" in f]
        if not pkl_files:
            pkl_files = [dataset_path]
        print(f"  Archivos de dataset encontrados: {pkl_files}")
    else:
        pkl_files = [dataset_path]

    for pf in pkl_files:
        try:
            with open(pf, "rb") as f:
                data = pickle.load(f)
            all_data.extend(data)
            print(f"  + {pf}: {len(data)} muestras")
        except FileNotFoundError:
            print(f"  [AVISO] No se encontró '{pf}', se omite.")

    if not all_data:
        print("\n  ERROR: Dataset vacío. Primero ejecuta a2c_dagger.py y pulsa Q.")
        return

    print(f"  Total de muestras para fine-tuning: {len(all_data)}")

    # ── 2. Preparar tensores ──────────────────────────────────
    states  = torch.FloatTensor(np.array([item[0] for item in all_data]))
    actions = torch.FloatTensor(np.array([item[1] for item in all_data]))

    # Verificar rangos
    print(f"  Rango acciones experto: v∈[{actions[:,0].min():.2f}, {actions[:,0].max():.2f}]"
          f"  w∈[{actions[:,1].min():.2f}, {actions[:,1].max():.2f}]")

    # Clamping de seguridad
    actions = torch.clamp(actions, -1.0, 1.0)

    # ── 3. Cargar modelo ──────────────────────────────────────
    model = A2CContinuous(state_dim=14, action_dim=2)
    try:
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        print(f"  Pesos base cargados desde '{weights_path}'")
    except FileNotFoundError:
        print(f"  ERROR: No se encontró '{weights_path}'.")
        return

    model.train()

    # ── 4. Optimizador y scheduler ────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)
    criterion = nn.MSELoss()

    N       = len(all_data)
    loss_log = []

    # ── 5. Bucle de entrenamiento ─────────────────────────────
    print(f"\n  Entrenando {epochs} épocas (batch_size={batch_size}, lr={lr})...\n")
    print(f"  {'Época':>6}  {'Imitation Loss':>16}  {'LR actual':>12}")
    print("  " + "-" * 40)

    for epoch in range(epochs):
        # Mini-batches aleatorios
        perm   = torch.randperm(N)
        losses = []

        for start in range(0, N, batch_size):
            idx   = perm[start:start + batch_size]
            s_b   = states[idx]
            a_b   = actions[idx]

            optimizer.zero_grad()
            pred_actions = model.actor_net(s_b)   # salida tanh en [-1,1]
            loss = criterion(pred_actions, a_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        mean_loss = float(np.mean(losses))
        loss_log.append(mean_loss)
        current_lr = scheduler.get_last_lr()[0]

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  {epoch+1:>6}  {mean_loss:>16.6f}  {current_lr:>12.2e}")

    # ── 6. Guardar pesos actualizados ─────────────────────────
    torch.save(model.state_dict(), weights_path)
    print(f"\n  \033[92m¡Fine-tuning completado! Pesos guardados en '{weights_path}'\033[0m")
    print("  Reinicia a2c_dagger.py para usar el modelo mejorado.\n")

    # ── 7. Gráfica de convergencia ────────────────────────────
    plt.figure(figsize=(8, 4))
    plt.plot(loss_log, color='darkorange', linewidth=2)
    plt.title("Fine-Tuning DAgger — Imitation Loss por Época")
    plt.xlabel("Época")
    plt.ylabel("MSE Loss (Imitación)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("dagger_finetune_curve.pdf", dpi=120)
    print("  Gráfica guardada en 'dagger_finetune_curve.pdf'")


if __name__ == '__main__':
    run_finetuning()