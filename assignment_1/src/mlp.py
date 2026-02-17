import numpy as np

class TwoLayerMLP:
    def __init__(self, M0, M1, M2, M3, activation='sigmoid', learning_rate=0.01, seed=42):
        """
        Args:
            M0, M1, M2, M3: Layer sizes (Neurons)
            activation: 'sigmoid' or 'relu'
        """
        self.activation_mode = activation
        self.lr = learning_rate
        np.random.seed(seed)
        
        # Initialization
        # W has shape (Current_Layer_Size, Previous_Layer_Size)
        
        # Scaling (He or Xavier)
        scale1 = np.sqrt(2.0/M0) if activation == 'relu' else np.sqrt(1.0/M0)
        scale2 = np.sqrt(2.0/M1) if activation == 'relu' else np.sqrt(1.0/M1)
        scale3 = np.sqrt(2.0/M2) if activation == 'relu' else np.sqrt(1.0/M2)
        
        self.W1 = np.random.randn(M1, M0) * scale1
        self.b1 = np.zeros((M1, 1))
        
        self.W2 = np.random.randn(M2, M1) * scale2
        self.b2 = np.zeros((M2, 1))
        
        self.W3 = np.random.randn(M3, M2) * scale3
        self.b3 = np.zeros((M3, 1))
    
    # Define internal-use helper functions
    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    
    def _sigmoid_prime(self, a):
        # Derivative w.r.t activation 'a'
        return a * (1 - a)
    
    def _relu(self, z):
        return np.maximum(0, z)
    
    def _relu_prime(self, z):
        return (z > 0).astype(float)
    
    def _softmax(self, z):
        # Softmax on columns (axis=0) because shape is (Features, Batch)
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    
    def forward(self, X_batch):
        """
        X_batch shape: (M0, N) -> Columns are examples
        """
        self.X0 = X_batch # Input Layer
        
        # Layer 1
        # A^(1) = W^(1)X^(0) + b^(1)
        self.A1 = self.W1 @ self.X0 + self.b1
        self.X1 = self._sigmoid(self.A1) if self.activation_mode == 'sigmoid' else self._relu(self.A1)
            
        # Layer 2
        # A^(2) = W^(2)X^(1) + b^(2)
        self.A2 = self.W2 @ self.X1 + self.b2
        self.X2 = self._sigmoid(self.A2) if self.activation_mode == 'sigmoid' else self._relu(self.A2)
            
        # Layer 3 (Output Layer)
        # A^(3) = W^(3)X^(2) + b^(3)
        self.A3 = self.W3 @ self.X2 + self.b3
        self.X3 = self._softmax(self.A3)
        
        return self.X3
    
    def backward(self, Y_one_hot):
        """
        Y_one_hot shape: (M3, N)
        """
        N = Y_one_hot.shape[1]
        
        # Layer 3 (output layer)
        delta3 = self.X3 - Y_one_hot
        
        # Gradients for Layer 3
        # dW^(3) = delta^(3) @ (x^(2))^T / N
        # db^(3) = sum(delta^(3)) / N
        dW3 = (delta3 @ self.X2.T) / N
        db3 = np.sum(delta3, axis=1, keepdims=True) / N
        
        # Layer 2
        # delta^(2) = ((W^(3))^T @ delta^(3)) * g'(s^(2))
        
        # 1. Backpropagate error through weights
        error_prop2 = self.W3.T @ delta3
        
        # 2. Compute derivative g'(s^(2))
        if self.activation_mode == 'sigmoid':
            # g'(s) = sigmoid(s) * (1 - sigmoid(s))
            # Since X2 = sigmoid(s2), we can use X2 * (1 - X2)
            g_prime_2 = self.X2 * (1 - self.X2)
        else:
            # g'(s) = 1 if s > 0 else 0
            # Strictly using s^(2) (self.A2)
            g_prime_2 = (self.A2 > 0).astype(float)
            
        # 3. Combine
        delta2 = error_prop2 * g_prime_2
        
        # Gradients for Layer 2
        # dW^(2) = delta^(2) @ (x^(1))^T / N
        dW2 = (delta2 @ self.X1.T) / N
        db2 = np.sum(delta2, axis=1, keepdims=True) / N
        
        # Layer 1
        # delta^(1) = ((W^(2))^T @ delta^(2)) * g'(s^(1))
        error_prop1 = self.W2.T @ delta2
        if self.activation_mode == 'sigmoid':
            g_prime_1 = self.X1 * (1 - self.X1)
        else:
            g_prime_1 = (self.A1 > 0).astype(float)
            
        delta1 = error_prop1 * g_prime_1
        
        # Gradients for Layer 1
        # dW^(1) = delta^(1) @ (x^(0))^T / N
        dW1 = (delta1 @ self.X0.T) / N
        db1 = np.sum(delta1, axis=1, keepdims=True) / N

        # Store Gradients for Analysis
        self.grads = {
            'dW1': np.mean(np.abs(dW1)), # Mean Absolute Magnitude
            'dW2': np.mean(np.abs(dW2))
        }
        
        # Update step (Gradient Descent)
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        
    def train(self, X_train, y_train, X_val, y_val, iterations=200):
        # NOTE: Inputs X must be (Features, Samples)
        # y_train must be integers (N,)
        
        history = {'train_acc': [], 'val_acc': [], 'train_loss': [],'grad_w1': [], 'grad_w2': []}
        M3 = self.W3.shape[0] # Output dim
        N_train = X_train.shape[1]
        
        for i in range(iterations):
            # Forward pass
            Y_hat = self.forward(X_train)
            
            # Loss & Acc
            correct_probs = Y_hat[y_train, np.arange(N_train)]
            loss = -np.mean(np.log(correct_probs + 1e-9))
            preds = np.argmax(Y_hat, axis=0)
            train_acc = np.mean(preds == y_train)
            
            # Backward pass
            # Create One-Hot Y (M3, N)
            Y_one_hot = np.zeros((M3, N_train))
            Y_one_hot[y_train, np.arange(N_train)] = 1
            self.backward(Y_one_hot)
            
            # Validation
            Y_val_hat = self.forward(X_val)
            val_preds = np.argmax(Y_val_hat, axis=0)
            val_acc = np.mean(val_preds == y_val)
            
            # Store history
            history['train_loss'].append(loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            # Store Gradients for Analysis
            history['grad_w1'].append(self.grads['dW1'])
            history['grad_w2'].append(self.grads['dW2'])
            
            if i % 20 == 0:
                print(f"Iter {i}: Loss {loss:.4f} | Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f}")
                
        return history