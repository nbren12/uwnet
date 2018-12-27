# Characterizing the time-sourced error of the NN

X^n+1 = X^n + dt * (F^n + G(X^n))

# The MSE of X^n+1 increases with dt, as the current state and forcing become less indicitve of the future state.

mses = []
dts = np.arange(0.125, 10 * min_time_step, min_time_step)
for dt in np.arange(0.125, 10 * min_time_step, min_time_step):
    mse = train_model(dt)
    mses.append(mse)
plt.scatter(dts, mses)

# This shows that the MSE of the model increases exponentially with dt.
