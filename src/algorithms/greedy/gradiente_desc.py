import numpy as np

def gradient_descent(x, y, learning_rate=0.01, iterations=1000):
    m = 0
    b = 0
    n = len(x)
    
    for _ in range(iterations):
        y_pred = m * x + b
        dm = (-2/n) * sum(x * (y - y_pred))
        db = (-2/n) * sum(y - y_pred)
        m = m - learning_rate * dm
        b = b - learning_rate * db
        
    return m, b

# Example usage
if __name__ == "__main__":
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([5, 7, 9, 11, 13])
    
    m, b = gradient_descent(x, y)
    print(f"Slope: {m}, Intercept: {b}")