# Convolutional Neural Networks
[Material](https://atcold.github.io/pytorch-Deep-Learning/en/week03/03/)

[Video](https://www.youtube.com/watch?v=FW5gFiJb-ig&list=PLLHTzKZzVU9eaEyErdV26ikyolxOsz6mq&index=6)

## Transformación de parámetros
- Si para una función $G(x,w)$, $w$ son sus parámetros y $w=H(u)$, es posible calcular $u$ con backpropagation.
    - Ejemplos:
        - *weight sharing* $\rightarrow w_1=w_2=u_1 \quad\&\quad w_3=w_4=u_2$
            - Usado en CNNs / detección de patrones repetitivos - *motifs* (por ejemplo: reconocer una forma en una imagen sin importar su posición)
        - *hypernetworks* $\rightarrow$ los pesos de una red son calculados a partir de la salida de otra red
            - Utilizado en casos que $\hat y$ sea una transformación de $x$ ($w$ son los parámetros de la transformación)

## Convolución discreta
Deslizar una función (kernel) sobre otra función y hallar la suma de sus productos en todo momento.

- Definición:
$$y_i=\sum_jw_jx_{(i-j})$$
- En la práctica (*cross-correlation*)
$$y_i=\sum_jw_jx_{(i+j)}$$
- En 2D
$$y_i=\sum_{kl}w_{kl}x_{(i+k,j+l)}$$

- Otros conceptos
    - Padding
    - Stride: tamaño de los pasos de cada paso de la convolución
    - Kernel shape / Kernel size: no siempre es cuadrado, puede tener diferentes tamaños

## Convolutional Neural Networks
- Primeras redes convolucionales: MNIST (Lecun 1989)
    - Shared weights
    - Avg. Pooling + Subsampling
    - Stride
    - Kernel 5x5
- Aprende representaciones **jerárquicas** en el entrenamiento
    - Primeras capas -> características de bajo nivel (rayas, bolas)
    - Siguientes capas -> características de alto nivel (formas cada vez más complejas, uniones de características sencillas)
- Arquitectura
    - Capas convolucionales
        - Normalización (Batch norm, ...)
            - Puede ir antes de pooling también
        - Filtro (Convolución)
        - No-linealidad (ReLU, Tanh, ...)
        - Pooling (Max. Pooling, Norma $L_p$)
    - Clasificador
        - Varias capas fully connected

## Usos de las redes convolucionales
- Señales multidimensionales
- Señales con alta correlación local
- Señales en las que sus características pueden aparecer en cualquier punto dentro de su espacio
- 2D CNNs:
    - Detección de objetos
    - Localización
    - Reconocimiento
- 3D CNNs:
    - Imágenes biomédicas
    - Video
    - Imágenes multiespectrales
