import matplotlib.pyplot as plt

from src.data import open_data


def get_data():
    ds = open_data('training')
    index = {'x': 0, 'y': 32, 'z': 20}
    location = ds.isel(**index)
    return location.to_dataframe()


def plot_twin(df, keys, labels, units, a):

    c1, c2 = ['b', 'y']

    x = df.index
    y1 = df[keys[0]]
    y2 = df[keys[1]]

    lab1, lab2 = labels

    l1, = a.plot(x, y1, c=c1)
    a_twin = a.twinx()
    l2, = a_twin.plot(x, y2, c=c2, alpha=.9)

    a_twin.set_ylabel(lab2, color=c2)
    a.set_ylabel(lab1, color=c1)


def plot_data(df):

    fig, (a, b, c) = plt.subplots(3, 1, sharex=True)
    plot_twin(df, ['QT', 'FQT'], ['QT', 'FQT'], [], a)
    plot_twin(df, ['SLI', 'FSLI'], ['SLI', 'FSLI'], [], b)
    c.plot(df.index, df.Prec)
    c.set_ylabel('mm/day')
    c.legend(['Precipitation'])


plot_data(get_data())
