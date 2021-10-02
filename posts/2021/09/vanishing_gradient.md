@def title = "Posts"
@def tags = ["syntax", "code"]
@def author = "Shuyi Jia"

\toc

## The Vanishing Gradient Problem
### Recurrent Neural Network
A **recurrent neural network** (RNN) is a class of artificial neural networks that is specifically designed to process sequential data. Some examples of sequential data include speeches, gene sequences, and historical stock prices.

Given an input sequence $\boldsymbol{x}_{1:T}=(\boldsymbol{x}_1,\cdots,\boldsymbol{x}_t,\cdots, \boldsymbol{x}_T)$, a RNN computes the hidden vector $\boldsymbol{h}_t$ for the current hidden layer:

$$
\boldsymbol{h}_t = f(\boldsymbol{h}_{t-1},\boldsymbol{x}_t),
$$

where $f(\cdot)$ is a nonlinear function, $\boldsymbol{h}_{t-1}$ the hidden vector calculated at the previous timestep and $\boldsymbol{x}_t$
the current input vector. $\boldsymbol{h}_t$ is also commonly referred to as the **hidden state**.

In the case of simple recurrent neural networks ([Elman, 1990](https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1402_1)),
the hidden state $\boldsymbol{h}_t$ is updated as follows:

\begin{align}
  \boldsymbol{h}_t &= f(U\boldsymbol{h}_{t-1} + W\boldsymbol{x}_t + \boldsymbol{b}),\\
  \hat{y}_t &= g(V\boldsymbol{h}_t),
\end{align}

where $U$ and $W$ are the recurrent and input weights respectively, $\boldsymbol{b}$ the bias vector and $\hat{y}_t$ the output at timestep $t$
by applying some function $g$ to the hidden state.

### Backpropagation Through Time
The network weights of RNN can be learnt through the usual gradient descent method.

Suppose our training data is $(\boldsymbol{x},\boldsymbol{y})$, where $\boldsymbol{x}_{1:T} = (\boldsymbol{x}_1,\cdots,\boldsymbol{x}_T)$
is a sequence of input vectors $\boldsymbol{x}_i$ and $\boldsymbol{y}_{1:T}=(y_1, \cdots,y_T)$ the corresponding true labels.
The loss function at timestep $t$ is defined as:

$$
\mathcal{L}_t = \mathcal{L}\left(y_t, \hat{y}_t\right),
$$

where $\mathcal{L}$ is a differentiable loss function such as the cross-entropy function.
The loss for the entire sequence is then

$$
\mathcal{L} = \sum^T_{t=1}\mathcal{L}_t.
$$

The partial derivative of $\mathcal{L}$ with respect to network weights $U$ is

$$
\frac{\partial \mathcal{L}}{\partial U}=\sum^T_{t=1}\frac{\partial \mathcal{L}_t}{\partial U}.
$$

To compute $\frac{\partial \mathcal{L}_t}{\partial U}$, we first have to look at $\frac{\partial \mathcal{L}_t}{\partial u_{ij}}$, where $u_{ij}$ 
is the $i$-th row, $j$-th column element of $U$:

$$
\frac{\partial \mathcal{L}_t}{\partial u_{ij}} = \sum^t_{k=1}\frac{\partial \mathcal{L}_t}{\partial \boldsymbol{h}_k} \frac{\partial \boldsymbol{h}_k}{\partial u_{ij}},
$$

for $1 \le k \le t$. In particular, for the first term in the summation, we have

\begin{align}
  \frac{\partial \mathcal{L}_t}{\partial \boldsymbol{h}_k} &= \frac{\partial \mathcal{L}_t}{\partial \boldsymbol{h}_t}\frac{\boldsymbol{h}_t}{\partial \boldsymbol{h}_k}\\
  &= \frac{\partial \mathcal{L}_t}{\partial \boldsymbol{h}_t} \prod^{t-1}_{\tau =k}\frac{\partial \boldsymbol{h}_{\tau + 1}}{\partial \boldsymbol{h}_\tau}\\
  &= \frac{\partial \mathcal{L}_t}{\partial \boldsymbol{h}_t} \prod^{t-1}_{\tau =k}\left(\text{diag}\left[f'(\boldsymbol{z}_\tau)\right]U\right).
\end{align}

### Exploding and Vanishing Gradients
For very long sequence length $T$, repeated multiplication of $U$ in Eq. $(5)$ can cause exponential growth or decay of backpropagated gradients.
In particular, exploding gradients are caused by eigenvalues of $U$ that are greater than $1$, while vanishing gradients are caused by eigenvalues
of $U$ that are less than $1$.

Another way of seeing this is to set $\gamma \cong ||\text{diag}\left[f'(\boldsymbol{z}_\tau)\right]U||$.

If $\gamma > 1$ and $t-k \rightarrow \infty$, then naturally we have $\gamma^{t-k}\rightarrow \infty$, therefore the exploding gradient problem.

Conversely, if $\gamma < 1$ and $t-k \rightarrow \infty$, we have $\gamma^{t-k}\rightarrow 0$, thus the vanishing gradient problem.

We can limit the exploding gradient problem by using gradient clipping. However, it is harder to obviate the vanishing gradient problem.

### Long Short-term Memory
Some solutions to the vanishing gradient problem include replacing sigmoid functions $\sigma$ with $\texttt{ReLU}$ and the use of gating mechanisms
such as **LSTM** and GRU. 

## References
1. **Carter N. Brown**, *Gradients for an RNN*, 2017, [[link]](https://github.com/go2carter/nn-learn/blob/master/grad-deriv-tex/rnn-grad-deriv.pdf).
2. **Dive Into Deep Learning**, *Backpropagation Through Time*, [[link]](https://d2l.ai/chapter_recurrent-neural-networks/bptt.html).
3. **Xipeng Qiu**, *Neural Networks and Deep Learning*, 2020 [[link]](https://nndl.github.io).
***