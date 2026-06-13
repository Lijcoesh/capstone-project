## Installation & Setup

### Important
Because of the fact we use a large dataset you'll need to download [GIT LFS (GIT Large File Storage)](https://git-lfs.com/) before you clone the repository.

### Fast Automation (Recommended)
An interactive setup script is provided for in the root directory to initialize the local Python virtual environment and fetch system dependencies:

```bash
cd /src/
chmod +x ./quick_setup.sh
./quick_setup.sh
source .venv/bin/activate
```

If you want to use an NVIDIA GPU for training, install PyTorch with CUDA support. This requires a compatible NVIDIA GPU and NVIDIA drivers installed on the host system.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify that PyTorch detects your GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Project Structure
```
- Project/            # Root
- Project/data/       # raw and processed data
- Project/notebooks/  # exploratory analysis
- Project/src/        # reusable scripts 
- Project/models/     # saved model files
- Project/results     # outputs, figures, metrics
- README.md
```
