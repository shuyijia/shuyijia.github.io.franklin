@def title = "Posts"
@def tags = ["syntax", "code"]
@def author = "Shuyi Jia"

\toc

## Recurrent Quantum Neural Networks
My notes/review on the paper [Recurrent Quantum Neural Networks](https://proceedings.neurips.cc/paper/2020/hash/0ec96be397dd6d3cf2fecb4a2d627c1c-Abstract.html).
### Motivation
Recurrent neural networks (RNNs), while being able to process sequential data, are notoriously difficult to train. At the same time, they also suffer from **vanishing** and **exploding** gradients. In 2017, Arjovsky, Shah and Bengio ([Arjovsky, 2016](http://proceedings.mlr.press/v48/arjovsky16.pdf)) proposed unitary evolution RNNs, which achieve then state-of-the-art performance using a novel parameterization of unitary weight matrices. Building on this idea of using unitary matrices, Bausch constructed the first quantum recurrent neural network using *paramterized quantum neurons*.

### Quantum Gates and Unitarity
The interactions of any quantum system can be described by a Hermitian operator $\vect{H}$, which, as a solution to the Schrödinger equation, creates the system's time evolution under the unitary map $\vect{U} = \exp\left(-it\vect{H}\right)$. 

Importantly, any quantum algorithm that is made up of a sequence of **unitary** quantum gates (matrices) is also unitary. Thus, a *parameterized quantum circuit* serves as a prime candidate for a unitary recurrent network.

A simple example of quantum circuits is the following 2-qubit controlled-$\texttt{NOT}$ gate:

\figenv{Controlled-NOT gate}{/figs/cnot.png}{object-fit: scale-down; max-width: 60%}

The symbol $\oplus$ is addition modulo two. The action of the gate can be described as follows. If the control qubit $\ket{A}$ is set to 0, then the target qubit $\ket{B}$ remains changed. If $\ket{A}$ is set to 1 instead, then the target qubit is flipped. In equations, we have

$$
\ket{00}\rightarrow \ket{00};\quad \ket{01}\rightarrow \ket{01};\quad \ket{10}\rightarrow \ket{11};\quad \ket{11}\rightarrow \ket{10}.
$$
Or more succinctly: $\ket{A,B} \rightarrow \ket{A, B\oplus A}$.

### Quantum Neuron
A classical neuron is a function that takes an input vector $\vect{x}$ and maps it to the output value $ a = \sigma(\vect{w}\cdot \vect{x} + \vect{b}) $, where $\sigma(\cdot)$ is a nonlinear activation function. Thus, the classical neuron consists of an affine transformation and a nonlinear mapping.

In order to achieve affine transformation and nonlinearity in quantum circuits, we will look at **rotations**. In particular, we have the simple single-qubit gate $\vect{R}(\theta)$:

\begin{align}
\vect{R}(\theta) = \begin{bmatrix}\cos\theta & -\sin\theta \\ \sin \theta & \cos\theta\end{bmatrix}
\end{align}

The simple quantum gate, when acted upon a qubit, is a rotation within the 2-dimensional space spanned by the computational basis vectors $\{\ket{0},\ket{1}\}$. We can thus construct the following quantum neuron (circuit):

\figenv{A first order quantum neuron}{/figs/qneuron.png}{width:75%;}

In the above circuit, $\ket{x}$ represents $n$ input qubits; the middle qubit $\ket{0}$ represents the ancilla qubit and the last qubit $\ket{0}$ is the output qubit.

If we raise the rotation to a controlled operation $\vect{cR}(i,\theta_i)$ conditioned on the $i$-th qubit of a state $\ket{x}$ for $x\in \{0,1\}^n$, one can derive the map:

\begin{align}\label{eq:map}
&\vect{R}(\theta_0)\vect{cR}(1,\theta_1)\cdots\vect{cR}(n,\theta_n)\ket{x}\ket{0}\\
&= \ket{x}\left( \cos(\eta)\ket{0} + \sin(\eta)\ket{1} \right),\\ 
&\text{where } \eta = \theta_0 + \sum^n_{i=1}\theta_ix_i. 
\end{align}

This corresponds to a rotation by an **affine transformation** of the basis vector $\ket{x}$ by a parameter vector $\vect{\theta}$.

@@explain
**Walkthrough Calculation of the Quantum Neuron**

Suppose $\ket{x} = \ket{0}$, that is, we have 1 condition qubit. We can also neglect $\vect{R}(\theta_0)$ in Eq. \eqref{eq:map}.

1. *Apply $\vect{cR}$ on Ancilla Qubit*
    - $\ket{1}\ket{0}\ket{0}\rightarrow \ket{1}\left( \cos\theta \ket{0} + \sin\theta \ket{1} \right)\ket{0}$
2. *Apply controlled $i\vect{Y}$ on Output Qubit*
    - $\ket{1}\left( \cos\theta \ket{0} + \sin\theta \ket{1} \right)\ket{0} \rightarrow \ket{1}\left( \cos\theta \ket{0}\ket{0} - \sin\theta \ket{1}\ket{1} \right)$
3. *Apply Undoing Rotation $\vect{R}(-\theta)$*
    - $\ket{1}\left( \cos\theta \ket{0}\ket{0} - \sin\theta \ket{1}\ket{1} \right) \rightarrow \ket{1}( \cos^2\theta \ket{00} + \cos\theta\sin\theta\ket{10} -\cos\theta\sin\theta\ket{11} + \sin^2\theta\ket{01})$
4. *Post-selection*
    - We post-select on $\ket{0}$, thus having $\cos^2\theta\ket{100} + \sin^2\theta\ket{101}$

By normalization requirement, the above can be written as
$$
\ket{10}\left(\frac{\cos^2\theta}{\sqrt{\cos^4\theta+\sin^4\theta}}\ket{0} + \frac{\sin^2\theta}{\sqrt{\cos^4\theta+\sin^4\theta}}\ket{1}\right),
$$
which corresponds to a rotation by angle $\phi$ on the output qubit, where
$$\label{eq:activationfunction}\phi = \arctan(\tan^2\theta).$$
@@

Eq. \eqref{eq:activationfunction} can be viewed as the quantum equivalent of the classical sigmoid function.

### QRNN Cell
Abstracting the quantum neuron away as $N$, Bausch constructed the following quantum recurrent neural network cell:

\figenv{QRNN cell (figure from [1])}{/figs/qrnncell.png}{width:auto;}

Similarly to classical RNN, this QRNN cell can be *unrolled* as

\figenv{Unrolled QRNN (figure from [1])}{/figs/unrolled.png}{width:auto;}

The inputs of QRNN are bitstrings while the outputs are post-selected qubits (thus deterministic outputs).

### Results
On the classification of MNIST hand-written digits, QRNN is able to achieve an accuracy of 94.6%.


## References
1. **Johannes Bausch**, [Recurrent Quantum Neural Networks](https://proceedings.neurips.cc/paper/2020/hash/0ec96be397dd6d3cf2fecb4a2d627c1c-Abstract.html), 2020.
2. **Cao, Yudong, Gian Giacomo Guerreschi, and Alán Aspuru-Guzik.**, [Quantum neuron: an elementary building block for machine learning on quantum computers](https://arxiv.org/pdf/1711.11240), 2017.
3. **Michael Nielsen and Issac Chuang**, [Quantum Computation and Quantum Information](https://aapt.scitation.org/doi/pdf/10.1119/1.1463744?casa_token=92G1ySEHMG8AAAAA:fjZzvdh1QeHjR4itYyItHIOWuLWdN3PH7VZGz_-V-LPtIZS4HrUMcEv4ew5p_bVeh-nCJcZydZ0g).

***