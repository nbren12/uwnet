# Physical constraints

It is possible to impose physical constraints such as non-negativity of
humidity and precipitation.

## Non-negativity of moisture

Humidity should be non-negative, but it should also be conserved. If negative
humidity points are present, then thresholding the humidity field at zero will
increase the total amount of water in the column. One simple method to fix this is to 

1. Threshold the humidity to lie above zero $ q_v^+ = max(0, q_v) $, and then
2. Diminish the humidity in the remaining points by a constant factor:
\begin{equation}
 q_v^{\text{non-negative}} = q_v^+ \frac{\langle q_v \rangle}{\langle q_v^+ \rangle} .
\end{equation}

The operator $ \langle \rangle $ is the mass-weighted vertical integration operator.
