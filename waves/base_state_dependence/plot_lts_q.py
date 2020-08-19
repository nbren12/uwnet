from heating_dependence_on_lts_moisture import open_data_with_lts_and_path
import matplotlib.pyplot as plt


ds = open_data_with_lts_and_path()
ds.lts[0].plot()
plt.savefig("lts.png")

plt.figure()
ds.path[0].plot()
plt.savefig("q.png")
