import numpy as np

def rosenbrock(x,y,a,b):
    return (a-x)**2 + b*(y-x**2)**2

def gradienteRosenbrock(vector, a=1, b=100):
    x, y = vector
    
    term1 = -2*(a-x)
    term2 = -4*b*x*(y-x**2)
    term3 = 2*b*(y-x**2)
    

    if not np.all(np.isfinite([term1, term2, term3])): 
        raise RuntimeWarning("Overflow en gradiente")
        
    return np.array([term1 + term2, term3])

def hessianoRosenbrock(vector, a=1, b=100):

    x, y = vector
    
    h11 = 2 - 4*b*(y - x**2) + 8*b*x**2
    h12 = -4*b*x
    h22 = 2*b

    if not np.all(np.isfinite([h11,h12,h22])): 
        raise RuntimeWarning("Overflow en Hessiano")
    
    return np.array([[h11, h12],
                    [h12, h22]])

def gradienteDescendente(gradiente: callable,
                        learningRate: float, 
                        puntoInicial: np.ndarray,
                        maxIter: int = 10000, 
                        toleranciaStep: float = 1e-4,
                        toleranciaGradiente: float = 1e-4, 
                        history = False) -> tuple:
    
    history_points = []
    x_k = np.asarray(puntoInicial).flatten()
    history_points.append(x_k)
    
    try:
        for iter_count in range(maxIter):
            gradiente_xk = np.asarray(gradiente(x_k)).flatten()
            
            if not np.all(np.isfinite(gradiente_xk)): 
                return x_k, f"Overflow en gradiente en iter {iter_count}", history_points
            
            norma_gradiente = np.linalg.norm(gradiente_xk)
            if norma_gradiente < toleranciaGradiente:
                return x_k, f"Convergencia por gradiente en iter {iter_count}", history_points
            
            paso = learningRate * gradiente_xk

            if not np.all(np.isfinite(paso)):
                return x_k, f"Overflow en paso en iter {iter_count}", history_points
                
            x_k1 = x_k - paso
            
            if not np.all(np.isfinite(x_k1)): 
                return x_k, f"Overflow en actualización en iter {iter_count}", history_points
            
            if np.linalg.norm(x_k - x_k1) < toleranciaStep:
                return x_k1, f"Convergencia por step en iter {iter_count}", history_points
            
            x_k = x_k1
            if history:
                history_points.append(x_k)
        
        return x_k, f"Máximo de iteraciones alcanzado", history_points
            
    except Exception as e:
        return x_k, f"Error: {str(e)}", history_points

def newtonRaphson(hessiano: callable, gradiente: callable,
                  puntoInicial: np.ndarray,
                  maxIter: int = 10000,
                  toleranciaStep: float = 1e-6,
                  toleranciaGrad: float = 1e-6,
                  history = False) -> tuple:

    x_k = np.asarray(puntoInicial).flatten()
    history_list = [x_k.copy()] if history else None

    for iter_count in range(maxIter):
        try:
            grad_k = gradiente(x_k)
            if np.linalg.norm(grad_k) < toleranciaGrad:
                return x_k, f"Convergencia gradiente en iter {iter_count}", history_list
            
            H_k = hessiano(x_k)

            try:
                delta_x = np.linalg.solve(H_k, -grad_k)
            except np.linalg.LinAlgError:
                return x_k, "Hessiano singular", history_list
                
            x_k1 = x_k + delta_x

            if np.linalg.norm(delta_x) < toleranciaStep:
                return x_k1, f"Convergencia step en iter {iter_count}", history_list
            
            if not np.all(np.isfinite(x_k1)):
                return x_k, "Overflow en actualización", history_list
                
            x_k = x_k1

            if history: 
                history_list.append(x_k.copy())
            
        except RuntimeWarning as e:
            return x_k, f"Error numérico: {str(e)}", history_list
            
    return x_k, f"Máximo de iteraciones: {maxIter}", history_list


def mse(y_true, y_pred):
    
    n = len(y_true)
    return (1/n) * np.linalg.norm(y_true - y_pred)**2

def mseGradiente(X, y, w):

    n = len(y)
    y_pred = X.dot(w)
    return (-2/n) * X.T.dot(y - y_pred)

def gradiente_descendente_regresion(X, y, eta, max_iter=100000, tol=1e-6):
    n = X.shape[0]
    w = np.zeros(X.shape[1])  # Empezamos con el vector de pesos en 0
 
    w_history = [w.copy()]
    train_errors = []
    
    for _ in range(max_iter):
        
        grad = mseGradiente(X, y, w)
        w_new = w - eta * grad
    
        w_history.append(w_new.copy())
        train_errors.append(mse(y, X.dot(w_new)))
        
        if np.linalg.norm(w_new - w) < tol:
            break
            
        w = w_new
        
    return w, w_history, train_errors


def wOptimo(X, y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)


def momentum_regresion(X, y, eta, beta=0.9, max_iter=100000, tol=1e-6):
    n = X.shape[0]
    w = np.zeros(X.shape[1])
    v = np.zeros_like(w)  # Velocidad inicial
    

    w_history = [w.copy()]
    train_errors = []
    
    for i in range(max_iter):
        grad = mseGradiente(X, y, w)
        v = beta * v + eta * grad
        w_new = w - v
        
        if not np.all(np.isfinite(w_new)):
            return w, w_history, train_errors, "Overflow en actualización"
        
        w = w_new
        
        current_error = mse(y, X.dot(w))
        train_errors.append(current_error)
        w_history.append(w.copy())
        
        # Verificamos convergencia
        if len(train_errors) > 1:
            if abs(train_errors[-1] - train_errors[-2]) < tol:
                return w, w_history, train_errors, f"Convergencia por error en iter {i}"
                
    return w, w_history, train_errors, "Máximo de iteraciones alcanzado"


def ridge_mse(y_true, y_pred, w, lambda_):
    """Calcula el MSE con regularización L2"""
    n = len(y_true)
    return (1/n) * np.linalg.norm(y_true - y_pred)**2 + lambda_ * np.linalg.norm(w)**2


def ridge_gradiente(X, y, w, lambda_):
    """Calcula el gradiente del MSE con regularización L2"""
    n = len(y)
    return (-2/n) * X.T.dot(y - X.dot(w)) + 2*lambda_*w


def ridge_exacta(X, y, lambda_):
    """Calcula la solución exacta de Ridge Regression usando SVD"""
    n = X.shape[0]
    return np.linalg.inv(X.T.dot(X) + n*lambda_*np.eye(X.shape[1])).dot(X.T).dot(y)


def ridge_gradiente_descendente(X, y, lambda_, eta, max_iter=100000, tol=1e-6):
    """Implementa Ridge Regression usando gradiente descendente"""
    n = X.shape[0]
    w = np.zeros(X.shape[1])
    
    # Historiales
    w_history = [w.copy()]
    train_errors = []
    
    for i in range(max_iter):
      
        grad = ridge_gradiente(X, y, w, lambda_)
        w_new = w - eta * grad
        
        current_error = ridge_mse(y, X.dot(w_new), w_new, lambda_)
        train_errors.append(current_error)
        w_history.append(w_new.copy())
 
        if np.linalg.norm(w_new - w) < tol:
            return w_new, w_history, train_errors, f"Convergencia en iter {i}"
            
        w = w_new
        
    return w, w_history, train_errors, "Máximo de iteraciones alcanzado"

