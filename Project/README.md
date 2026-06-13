## Installation & Setup

Install the Python dependencies:

```bash
pip install -r src/requirements.txt
```

If you want to use an NVIDIA GPU for training, install the CUDA build of PyTorch. This requires a compatible NVIDIA GPU and NVIDIA drivers installed on the host system.

```bash
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu126
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
