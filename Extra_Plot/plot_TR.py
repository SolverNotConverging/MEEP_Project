import pandas as pd
import matplotlib.pyplot as plt

files = ["1_5.csv", "3_0.csv", "6_0.csv", "9_0.csv"]
labels = ["1.5 mm", "3.0 mm", "6.0 mm", "9.0 mm"]
linestyles = ["-", "--", "-.", ":"]

plt.figure(figsize=(5, 3), dpi=300)

for f, label, ls in zip(files, labels, linestyles):

    # locate where the table begins
    with open(f) as file:
        lines = file.readlines()

    start = None
    for i, l in enumerate(lines):
        if l.startswith("Frequency"):
            start = i
            break

    df = pd.read_csv(f, skiprows=start)

    # convert Meep frequency → GHz
    freq_GHz = df["Frequency"] * 300

    T = df["T"]
    R = df["R"]

    # plot
    plt.plot(freq_GHz, T, color="green", linestyle=ls, label=f"T ({label})")
    plt.plot(freq_GHz, R, color="red", linestyle=ls, label=f"R ({label})")

plt.xlabel("Frequency (GHz)")
plt.ylabel("Transmission (T) and Reflection (R)")
plt.grid(True)
plt.xlim(16, 32)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig("plot_TR.png", dpi=300)
plt.savefig("plot_TR.svg", dpi=300)
plt.show()
