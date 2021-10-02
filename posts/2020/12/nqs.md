@def title = "NQS"
@def tags = ["syntax", "code"]
@def author = "Shuyi Jia"

\toc

## Neural-network Quantum States
### Monte Carlo Method
Random numbers can be used to evaluate definite integrals. Consider the following $1$-dimensional integral
$$
\theta = \int^1_0 g(x) dx.
$$
To compute the value of $\theta$, note that if $U$ is uniformly distributed over $(0,1)$, then we can express $\theta$ as
$$
\theta = E[g(U)]
$$
since $E[g(x)] = \int g(x)f(x)dx$. 

If $U_1,\ldots, U_k$ are independent uniform $(0,1)$ random variables, it follows that the random variables $g(U_1),\ldots,g(U_k)$ are independent and identically distributed random variables with mean $\theta$. By law of large numbers, we have
$$
\sum^k_{i=1} \frac{g(U_i)}{k} \rightarrow E[g(U)] = \theta,
$$
as $k\rightarrow \infty$.

Hence we can approximate $\theta$ by generating a large number of random numbers $u_i$ and taking as our approximation the average of $g(u_i)$. This approach is known as the *Monte Carlo* method.

### Importance Sampling
Simple Monte Carlo integration can suffer from low efficiency. For example, suppose we are interested in estimating the area under the curve for some normal distribution. Since most of the contributions to the integral come from around the mean, a simple Monte Carlo approach will spend a lot of time sampling the tails of the distribution. 

The idea of importance sampling is to increase the density of (random) points in regions of interest and hence improve the overall efficiency.

Let $\vect{X} = (X_1,\ldots, X_n)$ be a vector of $n$ random variables having a joint density function $f(\vect{x})=f(x_1,\ldots,x_n)$ and suppose that we are interested in estimating
$$
\theta = E[h(\vect{X})] = \int h(\vect{x})f(\vect{x})d\vect{x}.
$$
If $g(\vect{x})$ is another probability density such that $f(\vect{x}) = 0$ whenever $g(\vect{x} = 0)$, then we can express $\theta$ as

\begin{align}
\theta &= \int \frac{h(\vect{x}f(\vect{x}))}{g(\vect{x})}g(\vect{x})d\vect{x},\\
&= E_g\left[ \frac{h(\vect{X})f(\vect{X})}{g(\vect{X})} \right].
\end{align}

### Variational Monte Carlo
Variational Monte Carlo (VMC) is based on a direct application of Monte Carlo integration to explicitly correlated many-body wavefunctions. The variational principle of quantum mechanics states that the energy of a trial wavefunction will be greater than or equal to the energy of the exact wavefunction.

Suppose we first propose a trial wave function $\ket{\psi_T}$ with parameters $\theta$. The variational energy can then be calculated as
\begin{align}
E_V &= \frac{\bra{\psi_T}H\ket{\psi_T}}{\braket{\psi_T\vert\psi_T}},\label{eq:ve}\\
&= \frac{\int \psi_T^\ast (\vect{X})H\psi_T(\vect{X})d\vect{X}}{\int|\psi(\vect{X})|^2 d\vect{X}},
\end{align}
where $\vect{X}$ is an electronic configuration.

Eq. \eqref{eq:ve} can be evaluated using the Monte Carlo method described earlier. Specifically,
\begin{align}
E_V &= \frac{\int |\psi(\vect{X})|^2 \psi_T(\vect{X})^{-1} H \psi_T(\vect{X})d\vect{X}}{\int|\psi(\vect{X})|^2 d\vect{X}},\label{eq:ve_final}\\
&= \int \rho(\vect{X}) E_L (\vect{X})d\vect{X},
\end{align}
where
$$
\rho(\vect{X}) = \frac{|\psi(\vect{X})|^2}{\int|\psi(\vect{X})|^2 d\vect{X}}
$$
and
$$
E_L(\vect{X}) = \psi^{-1}_T(\vect{X})H \psi_T(\vect{X}). \label{eq:localenergy}
$$
Eq. \eqref{eq:localenergy} is known as the *local energy*. The variational energy $E_V$ may then be approximated by
$$
E_V \approx \frac{1}{M}\sum_{\vect{X}\in \{ \vect{X} \}_\rho } E_L(\vect{X}), \label{eq:ve_approx}
$$
where $\{ \vect{X} \}_\rho$ is a set of $M$ samples drawn from the distribution $\rho(\vect{X})$. These samples are obtained using the Metropolis-Hastings algorithm.

### Metropolis-Hastings Algorithm
The Metropolis-Hastings algorithm offers an easy way to draw samples from a distribution which is quite complicated. The algorithm constructs a sequence of samples $\{ \vect{X}_1, \vect{X}_2, \ldots, \vect{X}_n \}_\rho$ drawn from the distribution $\rho(\vect{X})$ by following a random walk:

1. Start Walker at random position $\vect{X}$.
2. Generate a new position $\vect{X}_\Gamma$ from some transition probability density function $\Gamma(\vect{X}\rightarrow\vect{X}_\Gamma)$.
3. Accept the new position with probability
$$
A(\vect{X}\rightarrow \vect{X}_\Gamma) = \min \left(1, \frac{\Gamma(\vect{X}_\Gamma \rightarrow \vect{X})\rho(\vect{X}_\Gamma)}{\Gamma(\vect{X} \rightarrow \vect{X}_\Gamma)\rho(\vect{X})}\right) \label{eq:metropolis_original}
$$
4. Add $\vect{X}$ to our sample set.
5. Loop steps $2\rightarrow 4$ until $n$ samples are generated.

In NQS, Eq. \eqref{eq:metropolis_original} is as follows:
$$
P\left(\vect{X}^{(k)}\rightarrow \vect{X}^{(k+1)}\right) = \min \left(1, \left\lvert\frac{\psi(\vect{X}^{(k+1)})}{\psi(\vect{X}^{(k)})}\right\rvert^2\right).
$$

### Machine Learning
With the basics out of the way, we can now describe the machine learning problem for NQS.

To minimize the energy (Eq. \eqref{eq:ve_approx}), we find the gradient of $E_L$ (Eq. \eqref{eq:localenergy}) with respect to the parameters $\theta$ in the trial wave function and we perform gradient descent update:
$$
\theta_{i+1} \leftarrow \theta_{i} - \alpha \frac{\partial E_L(\psi_T(\theta_i))}{\partial \theta_i}.
$$

### Restricted Boltzmann Machine
The last step is finding a suitable trial wave function representation in the learning. To this end, we use Restricted Boltzmann Machine (RBM). Our trial wave function is then:
$$
\psi_T(\vect{x},\vect{h}; \theta) = \exp \left( \vect{a}^T\vect{x} + \vect{b}^T\vect{h} + \vect{h}^T\vect{W}\vect{x} \right),
$$
where $\vect{x}$ is a particular spin configuration from $\vect{X}$ and also the **visible units**. The hidden units are represented by $\vect{h}$. The parameters $\theta$ consists of $\vect{a}$, $\vect{b}$ and $\vect{W}$.

Summing up the hidden units, we have
\begin{align}
\psi(\vect{x};\theta) &= \sum_{\vect{h}}\psi(\vect{x},\vect{h};\theta), \\
&= \exp \left(\vect{a}^T\vect{x}\right)\prod^M_{i=1}2\cosh \left( b_i + \sum^N_{j=1}W_{ij}x_j \right).
\end{align}
The above wave function would be the trial wave function in VMC.

## References
1. **Giuseppe Carleo**, **Matthias Troyer**, [Solving the quantum many-body problem with artificial neural networks](https://arxiv.org/pdf/1606.02318.pdf), 2017.
2. **M. Hutcheon**, [Variational Quantum Monte Carlo energetics of atoms and small molecules](https://web.stanford.edu/class/cme324/saad-schultz.pdf), 2018.
3. **Paul R. C. Kent**, [Variational Monte Carlo](https://web.ornl.gov/~kentpr/thesis/pkthnode20.html).
4. **Sheldon M. Ross**, [Simulation](https://www.elsevier.com/books/simulation/ross/978-0-12-415825-2), 5th edition, 2013.

***