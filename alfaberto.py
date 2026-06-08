import numpy as np

x = np.array([

    # A
    [1, 1, 1,
     1, 0, 1,
     1, 1, 1,
     1, 0, 1,
     1, 0, 1],

    # B
    [1, 1, 0,
     1, 0, 1,
     1, 1, 0,
     1, 0, 1,
     1, 1, 0],

    # C
    [1, 1, 1,
     1, 0, 0,
     1, 0, 0,
     1, 0, 0,
     1, 1, 1],

    # D
    [1, 1, 0,
     1, 0, 1,
     1, 0, 1,
     1, 0, 1,
     1, 1, 0],

    # E
    [1, 1, 1,
     1, 0, 0,
     1, 1, 0,
     1, 0, 0,
     1, 1, 1],

    # F
    [1, 1, 1,
     1, 0, 0,
     1, 1, 0,
     1, 0, 0,
     1, 0, 0],

    # G
    [1, 1, 1,
     1, 0, 0,
     1, 0, 1,
     1, 0, 1,
     1, 1, 1],

    # H
    [1, 0, 1,
     1, 0, 1,
     1, 1, 1,
     1, 0, 1,
     1, 0, 1],

    # I
    [1, 1, 1,
     0, 1, 0,
     0, 1, 0,
     0, 1, 0,
     1, 1, 1],

    # J
    [1, 1, 1,
     0, 0, 1,
     0, 0, 1,
     1, 0, 1,
     1, 1, 1],

    # K
    [1, 0, 1,
     1, 0, 1,
     1, 1, 0,
     1, 0, 1,
     1, 0, 1],

    # L
    [1, 0, 0,
     1, 0, 0,
     1, 0, 0,
     1, 0, 0,
     1, 1, 1],

    # M
    [1, 0, 1,
     1, 1, 1,
     1, 0, 1,
     1, 0, 1,
     1, 0, 1],

    # N
    [1, 0, 1,
     1, 1, 1,
     1, 1, 1,
     1, 0, 1,
     1, 0, 1],

    # O
    [1, 1, 1,
     1, 0, 1,
     1, 0, 1,
     1, 0, 1,
     1, 1, 1],

    # P
    [1, 1, 1,
     1, 0, 1,
     1, 1, 1,
     1, 0, 0,
     1, 0, 0],

    # Q
    [1, 1, 1,
     1, 0, 1,
     1, 0, 1,
     1, 1, 1,
     0, 0, 1],

    # R
    [1, 1, 1,
     1, 0, 1,
     1, 1, 1,
     1, 1, 0,
     1, 0, 1],

    # S
    [1, 1, 1,
     1, 0, 0,
     1, 1, 1,
     0, 0, 1,
     1, 1, 1],

    # T
    [1, 1, 1,
     0, 1, 0,
     0, 1, 0,
     0, 1, 0,
     0, 1, 0],

    # U
    [1, 0, 1,
     1, 0, 1,
     1, 0, 1,
     1, 0, 1,
     1, 1, 1],

    # V
    [1, 0, 1,
     1, 0, 1,
     1, 0, 1,
     0, 1, 0,
     0, 1, 0],

    # W
    [1, 0, 1,
     1, 0, 1,
     1, 0, 1,
     1, 1, 1,
     1, 0, 1],

    # X
    [1, 0, 1,
     0, 1, 0,
     0, 1, 0,
     0, 1, 0,
     1, 0, 1],

    # Y
    [1, 0, 1,
     0, 1, 0,
     0, 1, 0,
     0, 1, 0,
     0, 1, 0],

    # Z
    [1, 1, 1,
     0, 0, 1,
     0, 1, 0,
     1, 0, 0,
     1, 1, 1]
])

y = np.eye(26)

w1 = np.random.rand(15, 20) * 0.1
w2 = np.random.rand(20, 26) * 0.1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivada(x):
    return x * (1 - x)

#treinamento
for epoca in range(10000):

    camada_oculta = sigmoid(np.dot(x, w1))
    saida = sigmoid(np.dot(camada_oculta, w2))

    erro = y - saida

    d_saida = erro * sigmoid_derivada(saida)

    erro_oculta = d_saida.dot(w2.T)

    d_oculta = erro_oculta * sigmoid_derivada(camada_oculta)

    w2 += camada_oculta.T.dot(d_saida) * 0.1
    w1 += x.T.dot(d_oculta) * 0.1

    if epoca % 1000 == 0:
        print("Época:", epoca, "Erro:", np.mean(np.abs(erro)))

teste = np.array([
     1, 1, 1,
     1, 0, 1,
     1, 1, 1,
     1, 1, 0,
     1, 0, 1
])

resultado = sigmoid(np.dot(sigmoid(np.dot(teste, w1)), w2))

print(resultado)

indice = np.argmax(resultado)

letras = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

print("Letra:", letras[indice])