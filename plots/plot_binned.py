import altair as alt


def _to_plottable_dataframe(q2):

    df = q2.to_dataframe().reset_index()
    df["lts"] = df.lts_bins
    df["path"] = df.path_bins
    df = df.drop(["lts_bins", "path_bins"], axis=1)
    df = df.dropna()
    return df


def get_pressure_encoding():
    return alt.Y("p", axis=alt.Axis(title="p (hPa)"), scale=alt.Scale(domain=[1015, 0]))


def plot_line_by_key_altair(
    ds, key, title_fn=lambda x: "", cmap="viridis", c_sort="descending", c_title=""
):
    """Make line plots of Q1 and Q2 for different levels of a dataset

    Args:
        ds: dataset wit
    """

    if not c_title:
        c_title = key

    df = _to_plottable_dataframe(ds)

    z = get_pressure_encoding()

    color = alt.Color(
        key, scale=alt.Scale(scheme=cmap), sort=c_sort, legend=alt.Legend(title=c_title)
    )

    labels = [
        ("a", "QV", "g/kg", "Water Vapor"),
        ("b", "TABS", "K", "Temperature"),
        ("c", "Q1", "K/day", "Average Q₁"),
        ("d", "Q2", "g/kg/day", "Average Q₂"),
        ("e", "Q1NN", "K/day", "Q₁ Prediction"),
        ("f", "Q2NN", "g/kg/day", "Q₂ Prediction"),
    ]

    charts = []
    for letter, key, unit, label in labels:
        chart = (
            alt.Chart(df, width=150)
            .mark_line()
            .encode(alt.X(key, axis=alt.Axis(title=unit)), z, color, order="z")
            .properties(title=f"{letter}) {label}")
        )
        charts.append(chart)

    row1 = alt.hconcat(*charts[:3])
    row2 = alt.hconcat(*charts[3:])
    return alt.vconcat(row1, row2, title=title_fn(ds))
