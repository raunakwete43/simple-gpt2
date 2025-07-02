import torch
import tiktoken
from model import SimpleGPT2Module

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_samples = 5
max_tokens = 16

enc = tiktoken.get_encoding("gpt2")

text = "Hello, I'm GPT2 model,"


x = torch.tensor(enc.encode(text), dtype=torch.long).repeat(num_samples, 1)
# Internally handles the device placement

model = SimpleGPT2Module.load_from_checkpoint(
    "./checkpoints/lightning_logs/version_2/checkpoints/epoch=4-step=120.ckpt"
)

output = model.run_inference(x, max_tokens)

decoded_text = enc.decode_batch(output.detach().cpu().numpy().tolist())
print("Decoded text:")
for i, text in enumerate(decoded_text):
    print(f"Sample {i + 1}: {text}")
