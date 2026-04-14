import numpy as np
from src.mlp.mlp import MLP

# fake data
X = np.random.rand(100, 5)
y = (X.sum(axis=1) > 2.5).astype(int)

# init model
model = MLP(input_dim=5)

# train
for epoch in range(200):
    loss = model.train_step(X, y, learning_rate=0.1)

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# predict
preds = model.predict(X)

print("\nFinal loss:", loss)
print("First 10 predictions:", preds[:10])
print("First 10 true labels:", y[:10])