import os
import sys
import subprocess
import time

from src.config import CFG
from src.config import TRAIN_MODE, PROBA_TEST_STEP, PREDICT_PROBA_MODE

def run_command(command_str):
    """Executa um comando no terminal com o PYTHONPATH configurado."""
    print(f"\n{'=' * 60}")
    print(f" EXECUTING: {command_str}")
    print(f"{'=' * 60}\n")

    # Configura o ambiente para o Python encontrar a pasta raiz
    env = os.environ.copy()
    # Pega no caminho da pasta onde o script está (raiz do projeto)
    current_dir = os.getcwd()
    # Pega no caminho da pasta "mãe" (a que contém a pasta grid_security_project)
    parent_dir = os.path.dirname(current_dir)

    # Adicionamos AMBAS ao PYTHONPATH por segurança
    env["PYTHONPATH"] = current_dir + os.pathsep + parent_dir + os.pathsep + env.get("PYTHONPATH", "")

    start_time = time.time()

    process = subprocess.run([sys.executable] + command_str.split(), check=False, env=env)

    elapsed = time.time() - start_time

    if process.returncode != 0:
        print(f"\n ERROR: {command_str} failed with exit code {process.returncode}.")
        sys.exit(process.returncode)
    else:
        print(f"\n SUCCESS: {command_str} completed in {elapsed:.2f}s.")


def main():
    # Log inicial para debug - Se isto não aparecer, o erro é no import do CFG
    print(f" CONFIGURATION DETECTED: ENV={getattr(CFG, 'ENV_NAME', 'UNKNOWN')}")
    if TRAIN_MODE:
        print("\n[MODE] FULL TRAINING PIPELINE")

        # Passo 1: Forecasters
        if not os.path.exists(CFG.MODEL_MEAN_PATH):
            run_command("src/train_forecast.py")

        # Passo 2: ENN
        if not os.path.exists(CFG.MODEL_ENN_PATH):
            run_command("src/training_enn.py")

        # Passo 3: Dados e Classificador
        #if not os.path.exists(CFG.CSV_OUTPUT_PATH):

        run_command("src/collect_data.py")

        run_command("src/train_classifier.py")

    elif TEST_SINGLE_EPISODE:
        run_command("src/collect_data.py")

    elif PREDICT_PROBA_MODE:
        run_command("src/train_classifier.py")

    else:
        print("\n WARNING: No active mode selected in src/config.py.")


if __name__ == "__main__":
    main()