**Q-MoE-400** is a 400 million parameter **Sparse Mixture of Experts (MoE)** language model trained on the OpenWebText dataset. It was built using JAX/Flax on 8 TPU v3 chips to study the compute efficiency of sparse architectures compared to dense transformers.

This repository contains the **PyTorch** implementation for inference, along with scripts to convert original JAX checkpoints.

🔗 **Hugging Face Hub:** [QuarkML/Q-MoE-400](https://huggingface.co/QuarkML/Q-MoE-400)

## ⚡ Quick Usage (Hugging Face)

You can use this model directly with the Hugging Face `transformers` library. Since this model uses a custom architecture, `trust_remote_code=True` is required.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

path = "QuarkML/Q-MoE-400"

# Load tokenizer and model
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    path,
    trust_remote_code=True,
    dtype=torch.float16,  # optional but recommended for GPU
    device_map="auto"     # automatically maps to available device (CUDA/CPU)
)

# Generate text
inputs = tok("Neural network are ", return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

out = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.8
)

print(tok.decode(out[0], skip_special_tokens=True))
```

---

## 🛠️ Local Setup & Manual Inference

If you prefer to run the model using the provided standalone scripts (without the `AutoModel` API) or want to explore the architecture code, follow these steps.

### 1. Clone the repository
```bash
git clone https://github.com/sidharth72/Q-MoE-400.git
cd Q-MoE-400
```

### 2. Install Dependencies
```bash
pip install torch numpy transformers tiktoken
```

### 3. Download Checkpoints
To run the local [generate.py](cci:7://file:///c:/QuarkML/Projects/MoE/generate.py:0:0-0:0) script, you need to download the raw PyTorch checkpoint (`.pt` file).

👉 **[Download Checkpoints Here](https://huggingface.co/QuarkML/Q-MoE-400/tree/main/torch_checkpoints)**

Download `moe_torch_state_dict_90000.pt` (or the latest step) and place it in the root of the repository.

### 4. Run Generation
You can run the provided generation script directly:

```bash
python generate.py
```

Or use the model in your own script:

```python
import torch
from Q_MoE_400_torch import load_model_from_checkpoint
from generation_config import GenerationConfig, generate, get_tokenizer

# 1. Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model_from_checkpoint("./moe_torch_state_dict_90000.pt", device=device)

# 2. Setup Tokenizer
encode, decode = get_tokenizer()
prompt = "The future of AI is"
input_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

# 3. Generate
cfg = GenerationConfig(max_new_tokens=100, temperature=0.8, top_k=50)
out_ids = generate(model=model, input_ids=input_ids, cfg=cfg)
print(decode(out_ids[0].tolist()))
```

---

## 📂 Repository Structure

- [Q_MoE_400_torch.py](cci:7://file:///c:/QuarkML/Projects/MoE/Q_MoE_400_torch.py:0:0-0:0): The standalone PyTorch model definition (MoE Transformer, Router, Expert Layers).
- [generate.py](cci:7://file:///c:/QuarkML/Projects/MoE/generate.py:0:0-0:0): Script to run text generation using the downloaded checkpoint.
- [generation_config.py](cci:7://file:///c:/QuarkML/Projects/MoE/generation_config.py:0:0-0:0): Configuration for sampling parameters and tokenizer utilities.
- [jax_to_torch.py](cci:7://file:///c:/QuarkML/Projects/MoE/jax_to_torch.py:0:0-0:0): Utility to convert original Flax/Orbax checkpoints to PyTorch `state_dict`.
- [pretrain_Q_MoE_400m_JAX_TPU.ipynb](cci:7://file:///c:/QuarkML/Projects/MoE/pretrain_Q_MoE_400m_JAX_TPU.ipynb:0:0-0:0): The original training notebook used on Google Cloud TPUs.

---

## 🧠 Training (JAX/TPU)

The model was trained using JAX and Flax on 8 x TPU v3 chips. The training procedure, including the custom Pytree data loading and distributed training loop, is documented in the notebook:

📄 **[pretrain_Q_MoE_400m_JAX_TPU.ipynb](cci:7://file:///c:/QuarkML/Projects/MoE/pretrain_Q_MoE_400m_JAX_TPU.ipynb:0:0-0:0)**

### Checkpoint Conversion
If you train your own model using the JAX notebook, you can convert the resulting Orbax checkpoint to PyTorch using [jax_to_torch.py](cci:7://file:///c:/QuarkML/Projects/MoE/jax_to_torch.py:0:0-0:0):

```bash
python jax_to_torch.py
```
*(Ensure you modify the [ExportConfig](cci:2://file:///c:/QuarkML/Projects/MoE/jax_to_torch.py:293:0-297:23) in the script to point to your specific JAX checkpoint path.)*

---

## 📜 Citation

If you use this code or model in your research, please cite:

```bibtex
@misc{q-moe-400,
  author = {Quark Machine Learning},
  title = {Q-MoE-400: A Sparse Mixture of Experts Model},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/sidharth72/Q-MoE-400}}
}
```

## License
Apache 2.0
