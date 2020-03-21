import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 


def read_file(filename):
    df = pd.read_csv(filename, names=['x', 'y'])
    x = df['x'].values.tolist()
    b = df['y'].values.tolist()
    b = np.array(b).reshape(-1,1)
    return x, b

def init_design_matrix(x, n):
    row_size = len(x)
    col_size = n
    design_matrix = np.zeros((row_size, col_size))
    for col in range(n):
        for row in range(row_size):
            design_matrix[row][n-1-col] = x[row] ** col
    return design_matrix

def LU_decomposition(m):
    n = len(m)
    L = np.eye(n)
    U = np.eye(n)
    for k in range(n):
        U[k][k] = m[k][k]
        
        for i in range(k+1, n):
            L[i][k] = m[i][k] / U[k][k]
            U[k][i] = m[k][i]
        for i in range(k+1, n):
            for j in range(k+1, n):
                m[i][j] = m[i][j] - L[i][k]*U[k][j]
    return L, U

def inverse_matrix(L, U, rows_size):
    b = np.eye(rows_size)
    inverse_m = []
    for y in b:
        x = [0 for _ in range(len(y))]
        Ux = []
        for i in range(len(y)):
            tmp = y[i]
            for j in range(i):
                tmp = tmp - Ux[j]*L[i][j]
            Ux.append(tmp)
        for i in reversed(range(len(Ux))):
            tmp = Ux[i]
            for j in range(len(Ux) - i - 1):
                j = j + i + 1
                tmp = tmp - x[j]*U[i][j] 
            x[i] = tmp / U[i][i]
        inverse_m.append(x)
    inverse_m = np.array(inverse_m)
    inverse_m = inverse_m.T
    return inverse_m

def LSE(n, Lambda, filename):
    x_input, y_input = read_file(filename)
    A = init_design_matrix(x_input, n)
    target_matrix = np.matmul(A.T, A) + np.eye(n) * Lambda
    L, U = LU_decomposition(target_matrix)
    inverse_mat = inverse_matrix(L, U, n)
    
    AT_b = np.matmul(A.T, y_input)
    X = np.matmul(inverse_mat, AT_b)
    return X

def newton(n, filename):
    x, b = read_file(filename)
    A = init_design_matrix(x, n)
    x_old = np.zeros((n,1))
    for _ in range(1):
        AT = A.T
        hession = np.matmul(2*AT, A)
        L, U = LU_decomposition(hession)
        derivate = np.matmul(hession, x_old) - np.matmul(2*AT, b)

        inverse_hession = inverse_matrix(L, U, n)

        x_gradient = np.matmul(inverse_hession, derivate)
        x_new = x_old - x_gradient

        
        x_old = x_new
    
    return x_new

def show_result(LSE_coef, Newton_coef, n , Lambda, filename):
    df = pd.read_csv(filename, names=['x', 'y'])
    real_x = df['x'].values.tolist()
    real_y = df['y'].values.tolist()
    upper_bound = int(max(real_x))
    lower_bound = int(min(real_x)) 

    LSE_x = np.array(range(lower_bound, upper_bound))
    LSE_y = 0
    for i in range(n):
        LSE_y = LSE_y + LSE_x ** i * LSE_coef[n-1-i]

    A = init_design_matrix(real_x, n)
    LSE_pred = np.matmul(A, LSE_coef)
    comp_y = np.array(real_y).reshape(-1,1)
    LSE_error = np.square(LSE_pred - comp_y).sum()
    print('LSE:')
    print('LSE coefficient:', LSE_coef)
    print('Total error:', LSE_error)
    plt.subplot(2,1,1)
    plt.plot(real_x, real_y, 'ro')
    plt.plot(LSE_x, LSE_y)
    #plt.show()

    Newton_x = np.array(range(lower_bound, upper_bound))
    Newton_y = 0
    for i in range(n):
        Newton_y = Newton_y + Newton_x ** i * Newton_coef[n-1-i]

    A = init_design_matrix(real_x, n)
    Newton_pred = np.matmul(A, Newton_coef)
    comp_y = np.array(real_y).reshape(-1,1)
    Newton_error = np.square(Newton_pred - comp_y).sum()
    print('Newton method:')
    print('Newton coefficient:', Newton_coef)
    print('Total error:', Newton_error)
    plt.subplot(2,1,2)
    plt.plot(real_x, real_y, 'ro')
    plt.plot(Newton_x, Newton_y)
    plt.show()

def main():
    filename = 'testfile.txt'
    n = 3
    Lambda = 10000 
    LSE_coef = LSE(n, Lambda, filename)
    Newton_coef = newton(n, filename)
    show_result(LSE_coef, Newton_coef, n, Lambda, filename)


if __name__ == "__main__":
    main()