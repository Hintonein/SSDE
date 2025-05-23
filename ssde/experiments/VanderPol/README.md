## Test for Van der Pol equation


The `Van der Pol` oscillator was originally proposed to describe the automatic stabilization of current in vacuum tube oscillators. It was later widely used in the research of biology, electronics, mechanics and other fields, especially in the study of nonlinear dynamic systems and periodic behavior. These systems often exhibit complex dynamic behaviors, such as chaos and multistability, which are of great scientific significance for understanding nonlinear systems and have the following forms:

$$  
\frac{\partial^2 u}{\partial t^2} - \mu (1-u^2) \frac{\partial u}{\partial t} + u = 0
$$

Considering $\mu = 0$，the equation degenerates into a linear oscillator equation, and a closed-form solution can be obtained:

$$
\frac{\partial^2 u}{\partial t^2} + u = 0\\
BCs: u(0) = 0, u(1) = 1 \\
Sol: u = \sin(\pi x)
$$

The log file is as follows:

1. `log/haros.log`: Find the closed-form solution without genetic programming.
2. `log/haros_gp.log`: Find the closed-form solution with genetic programming.
3. `log/vanderpol_gp.log`: Find the closed-form solution with genetic programming.
4. `log/vanderpol.log`: Find the closed-form solution without genetic programming.
5. 