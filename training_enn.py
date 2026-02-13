import numpy as np
import ray
import torch
import torch.optim as optim
import grid2op
from tqdm import tqdm
from grid2op.Reward import L2RPNReward
from lightsim2grid import LightSimBackend
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from config import CFG, AGENT_DIR, ENN_DATA_DIR
from enn_models import EvidentialNetwork, edl_loss

from pathlib import Path

# Try to import the Senior Agent
try:
    from curriculumagent.senior.senior_student import Senior
except ImportError:
    print("\n CRITICAL: Could not import 'Senior' network36.")
    print(f"Please ensure 'curriculumagent' folder is inside {AGENT_DIR}")
    exit(1)


def generate_data() -> None:
    """Simulates the Senior Agent to generate Training Data for ENN."""
    print(f"\n[GEN] Starting Data Generation for ENN ({CFG.ENV_NAME})...")

    enn_data_dir = Path(ENN_DATA_DIR)
    agent_path = Path(CFG.AGENT_PATH)
    # Check if data exists
    train_file = enn_data_dir / "senior_expert_train.npz"
    if train_file.exists():
        print("[INFO] Data already exists. Skipping generation.")
        return

    env = grid2op.make(CFG.ENV_NAME, reward_class=L2RPNReward, backend=LightSimBackend())

    # Setup Agent Paths
    checkpoint_path = agent_path / "senior" / "checkpoint_000250"
    actions_file = agent_path / "actions" / "actions.npy"

    # Ensure actions file exists
    if not actions_file.exists():
        actions_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(actions_file, env.action_space.to_vect())

    # Load Agent
    if not ray.is_initialized():
        ray.init(num_gpus=0, ignore_reinit_error=True)

    senior_agent = Senior(
        env_path=CFG.ENV_NAME,
        action_space_path=[str(actions_file)],
        model_path=agent_path / "junior",
        ckpt_save_path=None,
        run_with_tf=True,
        num_workers=1
    )
    senior_agent.restore(str(checkpoint_path))
    ppo_agent = senior_agent.ppo

    # Data Collection Loop
    all_obs = []
    all_acts = []
    conf = CFG.ENN_TRAIN_CONFIG

    # Type casting for config values
    num_episodes = int(conf['num_episodes'])
    max_steps = int(conf['max_steps'])

    for _ in tqdm(range(num_episodes), desc="Generating Episodes"):
        obs = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            obs_vect = obs.to_vect()

            # Get Action from PPO Agent (Integer index)
            action_idx = ppo_agent.compute_single_action(obs_vect, explore=False)

            all_obs.append(obs_vect)
            all_acts.append(action_idx)

            # Step in environment (using empty action to speed up simulation)
            obs, _, done, _ = env.step(env.action_space())
            steps += 1

    ray.shutdown()

    # Save Data
    X = np.array(all_obs, dtype=np.float32)
    y = np.array(all_acts, dtype=np.int64)

    # Shuffle and Split
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    n_train = int(len(X) * 0.8)

    np.savez(enn_data_dir / f"senior_expert_train_{CFG.ENV_NAME}.npz", s=X[:n_train], a=y[:n_train])
    np.savez(enn_data_dir / f"senior_expert_val_{CFG.ENV_NAME}.npz", s=X[n_train:], a=y[n_train:])
    print("[GEN] Data saved.")


def train_enn() -> None:
    """Trains the Evidential Neural Network."""
    conf = CFG.ENN_TRAIN_CONFIG
    lamb_val = float(conf['edl_lambda'])

    print(f"\n[TRAIN] Training ENN with Lambda={lamb_val}...")

    enn_data_dir = Path(ENN_DATA_DIR)
    # Load Data
    d_train = np.load(enn_data_dir / f"senior_expert_train_{CFG.ENV_NAME}.npz")
    d_val = np.load(enn_data_dir / f"senior_expert_val_{CFG.ENV_NAME}.npz")

    # Normalize Input
    scaler = StandardScaler()
    s_train = scaler.fit_transform(d_train["s"])
    s_val = scaler.transform(d_val["s"])

    # Create DataLoaders
    batch_size = int(conf['batch_size'])
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(s_train), torch.LongTensor(d_train["a"])),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(s_val), torch.LongTensor(d_val["a"])),
                            batch_size=batch_size)

    # Initialize Model
    device = torch.device(str(conf['device']))
    model = EvidentialNetwork(
        input_dim=CFG.ENN_INPUT_DIM,
        num_classes=CFG.ENN_NUM_CLASSES,
        hidden_dim=CFG.ENN_HIDDEN_DIM,
        dropout=float(conf['dropout'])
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=float(conf['max_lr']), weight_decay=float(conf['weight_decay']))

    epochs = int(conf['epochs'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=float(conf['max_lr']),
                                                    steps_per_epoch=len(train_loader), epochs=epochs)

    best_loss = float('inf')
    annealing_steps = int(conf['kl_annealing_steps'])

    # Training Loop
    for epoch in range(epochs):
        model.train()
        for batch_s, batch_a in train_loader:
            batch_s, batch_a = batch_s.to(device), batch_a.to(device).view(-1)
            optimizer.zero_grad()

            # Forward Pass (Get Alphas)
            alpha = model(batch_s)

            # Calculate EDL Loss
            loss = edl_loss(None, batch_a, alpha, epoch, CFG.ENN_NUM_CLASSES,
                            annealing_steps, device, lamb=lamb_val)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Validation Step
        if (epoch + 1) % 10 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for v_s, v_a in val_loader:
                    v_s, v_a = v_s.to(device), v_a.to(device).view(-1)
                    alpha = model(v_s)
                    val_loss += edl_loss(None, v_a, alpha, epoch, CFG.ENN_NUM_CLASSES,
                                         annealing_steps, device, lamb=lamb_val).item()

            avg_val = val_loss / len(val_loader)
            print(f"Ep {epoch + 1} | Val Loss: {avg_val:.4f}")

            if avg_val < best_loss:
                best_loss = avg_val
                torch.save(model.state_dict(), CFG.MODEL_ENN_PATH)

    print(f"[SAVE] ENN Model saved to {CFG.MODEL_ENN_PATH}")


if __name__ == "__main__":
    generate_data()
    train_enn()