This code allows to simulate a simplified handy model [https://doi.org/10.1016/j.ecolecon.2014.02.014] with random noise (parametrized by the random number $R$):

 a) Commoners: 
 $$\displaystyle \frac{dx_C}{dt} = \beta x_C - \alpha  x_C + \epsilon R x_C ,$$

 b) Nature:
 $$\displaystyle \frac{dy}{dt} = \gamma y \left ( \lambda - y \right ) - \delta x_Cy ,$$
 
 c) Wealth: 
 $$\displaystyle \frac{dw}{dt} = \delta x_Cy - C_C \left ( x_C, w \right ) - C_E \left ( x_E, w \right ) .$$

you can choose : 
the number of tries on line 3, 
the number of years on line 4,
and the value $\epsilon$ (amplitude of random number) on line 9.

