
# Backpropagation
[Material](https://atcold.github.io/pytorch-Deep-Learning/en/week02/02/)

[Video](https://www.youtube.com/watch?v=d9vdh3b787Y&list=PLLHTzKZzVU9eaEyErdV26ikyolxOsz6mq&index=3)
## Modelos parametrzados
$\hat y=G(x,w)$

Donde $x$ es la entrada y $w$ son los *parámetros* del modelo
- Generalmente $w$ es implícito (no se usa como entrada al modelo, es un parámetro interno)
- Ej. Regresión lineal: $\hat y=\sum_i w_ix_i$

## Función costo
Da una diferencia entre $y$ (valor esperado) y $\hat y$ (valor obtenido en $G$)

Para minimizar (obtener los valores de $w$ que minimizan el costo): ***Gradient descent***
$$w\leftarrow w - \eta \nabla C$$
- Donde: 
    - $w$ son los parámetros de $\hat y$
    - $C$ es la función costo
    - $\nabla C$ es el gradiente de la función costo $C$ en función de los parámetros $w$
    - $\eta$ es la tasa de aprendizaje.
- Puede ser
    - Batch GD
    - Stochastic GD
    - Mini-batch GD
- Solo usado para funciones $G$ diferenciables.
    - Otros métodos de optimización se usan para funciones no diferenciables.

## Red neuronal
- Unión de capas "función lineal - función no lineal"
- Generalmente función lineal es weighted sum y función no lineal es sencilla (tanh, relu)
- Gradiente se calcula con *Backpropagation* (Regla de la cadena)
    - En forma matricial: Jacobiano
    - Para *Autograd* (pytorch, tf) se utilizan grafos (con multiplicaciones matriciales) construidos para calcular automáticamente el gradiente de las redes.
        - Se pueden usar grafos para hallar derividas de cualquier función representable como un grafo acíclico dirigido (DAG)
- Bloques lineales (capa $k$): $s_{k+1}=w_kx_k$ (en forma matricial)
- Bloques no lineales: $z_k$=$h(s_k)$

### Backprop en la práctica
- Se usa ReLU (en vez de tanh o sigmoid). Principalmnete para para redes con muchas capas.
    - Debido a que es scale-invariant
- Función de costo es cross-entropy (con logsoftmax o similar NO SOFTMAX)
- Mini-batch gradient descent en el entrenamiento
    - Optimizador Adam con momento
    - Barajar las muestras de entrenamiento
    - Cada batch **debe** tener varias categorías
- Normalizar el set de entrenamiento
- La tasa de aprendizaje disminuye progresivamente
- Regularización L1 y L2 un poco en las funciones lineales (y solo después de algunos epochs)
    - No siempre
- Usar dropout
- Trucos de inicialización de pesos
- **(Mucho más en "Neural Networks, Tricks of the Trade (2012) o Efficient Backprop (1998))**