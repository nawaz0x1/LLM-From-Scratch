import torch
from data_loader import create_dataloader_v1

# Constents
vocab_size = 50257
output_dim = 256
max_length = 4
context_length = max_length

# Loading raw text
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Creating dataloader
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("Token IDs:\n", inputs)
print("Inputs shape:\n", inputs.shape)



# Token Embedding layer
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

# Positional Embedding layer 
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim) 
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape )

# Combining token and positional embeddings
input_pos_embeddings = token_embeddings + pos_embeddings


print(f"Token Embeddings shape: {token_embeddings.shape}")
print(f"Positional Embeddings shape: {pos_embeddings.shape}")
print(f"Input + Positional Embeddings shape: {input_pos_embeddings.shape}")