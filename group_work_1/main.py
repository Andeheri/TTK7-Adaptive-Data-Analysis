
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np


def main():
    filepath = Path("group_work_1/Signal1_2018.csv")
    fs = 100  # Sampling frequency

    if not filepath.exists():
        print(f"File {filepath} does not exist.")
        return
    
    signal = np.loadtxt(filepath, delimiter=",")
    t = np.arange(len(signal)) / fs  # Time vector

    plt.plot(t, signal * 1e6)  # Convert to microvolts
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [$\\mu V$]")
    plt.title("Signal 1")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()