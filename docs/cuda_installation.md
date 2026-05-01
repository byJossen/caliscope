# Installing Caliscope With CUDA

Use this install path if you want ONNX tracking/reconstruction to run on an NVIDIA GPU instead of CPU.

## 1. Check NVIDIA Driver

```powershell
nvidia-smi
```

If this fails, install or update your NVIDIA driver first.

## 2. Install Caliscope

```powershell
uv venv --python 3.12
.\.venv\Scripts\activate

uv pip install caliscope[gui]
uv pip uninstall onnxruntime
uv pip install onnxruntime-gpu
```

## 3. Verify CUDA

```powershell
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

CUDA is working if you see:

```text
CUDAExecutionProvider
```

Example:

```text
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

## 4. Launch Caliscope

```powershell
caliscope
```

In the Reconstruction tab, select an ONNX tracker. The GUI should show:

```text
Inference: CUDA
```

## If It Still Shows CPU

Make sure both packages are not installed at the same time:

```powershell
uv pip list | Select-String -Pattern "onnxruntime"
```

If both `onnxruntime` and `onnxruntime-gpu` appear, run:

```powershell
uv pip uninstall onnxruntime
uv pip install --force-reinstall onnxruntime-gpu
```

Then verify again.

See the official [ONNX Runtime CUDA documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) for CUDA/cuDNN compatibility details.
