# Linear Support Vector Machine With the Squared Hinge Loss

- Statistical Machine Learning For Data Scientists Code Release Practice

$$ \nabla \ F(\beta) = -\frac{2}{n}\sum_{i=1}^n u_{i} v_{i} + 2\lambda\beta $$ 
where $ u_{i} = max(0, 1 - y_{i}x_{i}^T\beta)$ and  $v_{i} = x_{i}y_{i}$
