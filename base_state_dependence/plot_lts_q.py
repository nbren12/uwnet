from heating_dependence_on_lts import open_data_with_lts_and_path


ds = open_data_with_lts_and_path()
ds.lts[0].plot()
plt.savefig("lts.png")

ds.q[0].plot()
plt.savefig("q.png")
