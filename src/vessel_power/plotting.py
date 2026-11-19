import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def create_figures(z, metrics, folder):
    folder.mkdir(exist_ok=True)
    h = metrics[
        metrics.split.eq("evaluate") & metrics.month.ne("all") & metrics.regime.eq("active")
    ]
    table = h.pivot(index="month", columns="method", values="mae")
    ax = table.plot.bar(figsize=(10, 5), rot=0)
    ax.set(
        xlabel="Historical month",
        ylabel="MAE against eligible recorded proxies (kW)",
        title="Active-operation errors; physical accuracy unverified",
    )
    ax.legend(["Historical offset (72h)", "Rolling cubic (72h)", "Static cubic"], fontsize=8)
    ax.figure.tight_layout()
    ax.figure.savefig(folder / "monthly_errors.png", dpi=180)
    plt.close(ax.figure)
    h = z[z.proxy_error_allowed & z.split.eq("evaluate") & z.inference_regime.eq("active")].copy()
    h["day"] = (h.index - pd.Timedelta(minutes=1)).floor("D")
    fields = ["power", "static_rpm_cubic", "historical_offset_72h", "rolling_rpm_cubic_72h"]
    daily = (
        h.groupby("day")[fields]
        .mean()
        .reindex(pd.date_range("2024-09-01", "2024-12-31", freq="D", tz="UTC"))
    )
    ax = daily.plot(figsize=(11, 5), marker=".", linewidth=1)
    ax.set(
        xlabel="Covered UTC day",
        ylabel="Mean active power (kW)",
        title="Eligible recorded proxies and conditional historical estimates",
    )
    ax.legend(
        ["Recorded proxy", "Static cubic", "Historical offset (72h)", "Rolling cubic (72h)"],
        fontsize=8,
    )
    ax.figure.tight_layout()
    ax.figure.savefig(folder / "daily_power.png", dpi=180)
    plt.close(ax.figure)
