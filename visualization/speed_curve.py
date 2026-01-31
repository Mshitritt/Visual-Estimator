import csv
import matplotlib.pyplot as plt

# CSV_PATH = r"../outputs/speeds_LK.csv"
CSV_PATH = r"../outputs/speeds_ORB.csv"

t = []
speed = []
ok = []

with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        t.append(float(row["t_sec"]))
        speed.append(float(row["speed_mps"]))
        ok.append(int(row["ok"]))

# Optionally: hide invalid frames (ok==0) instead of plotting zeros
t_ok = [ti for ti, oi in zip(t, ok) if oi == 1]
s_ok = [si for si, oi in zip(speed, ok) if oi == 1]

plt.figure()
plt.plot(t_ok, s_ok)  # or plot(t, speed) if you want to see the zeros too
plt.xlabel("Time (s)")
plt.ylabel("Speed (m/s)")
plt.title("Speed curve - ORB")
plt.grid(True)
plt.show()
