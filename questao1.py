import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=10):
        self.weights = np.random.rand(input_size + 1) * 0.01 
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        x = np.insert(x, 0, 1) 
        weighted_sum = np.dot(self.weights, x)
        return self.activation_function(weighted_sum)

    def train(self, X, y):
        epoch = 0
        while True:
            error_exists = False
            for xi, target in zip(X, y):
                xi = np.insert(xi, 0, 1) 
                weighted_sum = np.dot(self.weights, xi)
                prediction = self.activation_function(weighted_sum)
                if prediction != target:
                    error = target - prediction
                    self.weights += self.learning_rate * error * xi
                    error_exists = True
            epoch += 1
            if not error_exists or epoch >= self.epochs:
                break

        return self.weights

# Tabela Verdade para AND
X_and = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_and = [0, 0, 0, 1]

# Tabela Verdade para OR
X_or = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_or = [0, 1, 1, 1]

# Treinar o Perceptron para AND
perceptron_and = Perceptron(input_size=2)
weights_and = perceptron_and.train(X_and, y_and)
print(f'Pesos finais para AND: {weights_and}')

# Treinar o Perceptron para OR
perceptron_or = Perceptron(input_size=2)
weights_or = perceptron_or.train(X_or, y_or)
print(f'Pesos finais para OR: {weights_or}')

# Testar o Perceptron para AND
print("Resultados da operação AND:")
for xi, target in zip(X_and, y_and):
    prediction = perceptron_and.predict(xi)
    print(f'Entrada: {xi}, Esperado: {target}, Predição: {prediction}')

# Testar o Perceptron para OR
print("Resultados da operação OR:")
for xi, target in zip(X_or, y_or):
    prediction = perceptron_or.predict(xi)
    print(f'Entrada: {xi}, Esperado: {target}, Predição: {prediction}')