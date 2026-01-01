import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import wandb

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
data_path = 'tinyshakespeare.txt'
if not os.path.exists(data_path):
    print(f"Error: {data_path} not found.")
    exit(1)

with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Character mapping
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# Train/Val split (90/10)
n = len(text)
train_data = text[:int(n*0.9)]
val_data = text[int(n*0.9):]

# Hyperparameters
config = {
    "hidden_size": 256,
    "seq_length": 100,
    "learning_rate": 0.003,
    "epochs": 100,
    "batch_size": 128
}

# Initialize wandb
wandb.init(project="shakespeare-rnn", config=config)
config = wandb.config

# Simple RNN Model (Manual Implementation)
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Manual RNN weights
        self.i2h = nn.Linear(hidden_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        batch_size, seq_len = x.shape
        x = self.embedding(x)
        
        h_t = hidden[0]
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t = torch.tanh(self.i2h(x_t) + self.h2h(h_t))
            outputs.append(h_t.unsqueeze(1))
            
        out = torch.cat(outputs, dim=1)
        out = self.fc(out)
        
        return out, h_t.unsqueeze(0)

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size).to(device)

model = SimpleRNN(vocab_size, config.hidden_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# Prepare data for training
def get_batches(data, batch_size, seq_length):
    chars_encoded = np.array([char_to_ix[ch] for ch in data])
    num_chars = len(chars_encoded)
    num_steps = (num_chars - 1) // seq_length
    
    data_len = num_steps * seq_length
    x = chars_encoded[:data_len]
    y = chars_encoded[1:data_len+1]
    
    x = x.reshape(-1, seq_length)
    y = y.reshape(-1, seq_length)
    
    for i in range(0, x.shape[0], batch_size):
        yield torch.tensor(x[i:i+batch_size]).to(device), torch.tensor(y[i:i+batch_size]).to(device)

# Validation function
def validate(model, data, batch_size, seq_length):
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for x, y in get_batches(data, batch_size, seq_length):
            if x.shape[0] != batch_size: continue
            hidden = model.init_hidden(batch_size)
            output, hidden = model(x, hidden)
            loss = criterion(output.view(-1, vocab_size), y.view(-1))
            total_loss += loss.item()
            num_batches += 1
    model.train()
    return total_loss / num_batches if num_batches > 0 else 0

# Text generation function
def generate(model, start_str='First Citizen:', length=200):
    model.eval()
    with torch.no_grad():
        chars_gen = [char_to_ix[ch] for ch in start_str]
        input_seq = torch.tensor(chars_gen).unsqueeze(0).to(device)
        hidden = model.init_hidden(1)
        
        _, hidden = model(input_seq[:, :-1], hidden)
        
        result = start_str
        curr_char = input_seq[:, -1:]
        
        for _ in range(length):
            output, hidden = model(curr_char, hidden)
            probs = torch.softmax(output[0, -1], dim=0)
            next_ix = torch.multinomial(probs, 1).item()
            result += ix_to_char[next_ix]
            curr_char = torch.tensor([[next_ix]]).to(device)
            
    model.train()
    return result

# Training loop
print("Starting training...")
for epoch in range(config.epochs):
    total_train_loss = 0
    num_train_batches = 0
    
    for x, y in get_batches(train_data, config.batch_size, config.seq_length):
        if x.shape[0] != config.batch_size: continue
        
        optimizer.zero_grad()
        hidden = model.init_hidden(config.batch_size) 
        
        output, hidden = model(x, hidden)
        loss = criterion(output.view(-1, vocab_size), y.view(-1))
        
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        num_train_batches += 1
        
    avg_train_loss = total_train_loss / num_train_batches
    avg_val_loss = validate(model, val_data, config.batch_size, config.seq_length)
    
    # Log to wandb
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss
    })
    
    print(f"Epoch {epoch+1}/{config.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    # if (epoch + 1) % 1 == 0:
    #     sample = generate(model, length=100)
    #     print("--- Sample generation ---")
    #     print(sample)
    #     print("-------------------------")
    #     wandb.log({"samples": wandb.Html(f"<pre>{sample}</pre>")})

print("Training finished.")
wandb.finish()

# Save the model
torch.save(model.state_dict(), 'rnn_shakespeare.pth')
print("Model saved to rnn_shakespeare.pth")
