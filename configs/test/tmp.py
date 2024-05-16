import pickle

with open("search_result_figure.pkl", "rb") as f:
    a = pickle.load(f)

print("Done.")

figure_dict = a
for k, v in figure_dict.items():
    s = ""
    batch_size, step_times = v
    s+= f"\n\n======================= Experiment for {k}:\n"
    ss = []
    for b, t in zip(batch_size, step_times):
        ss.append([b, f"Batch={b:3d}: {t/b:7.3f} second/sample, {b/t:7.3f} samples/second, {t:7.3f}s"])
    ss = sorted(ss,key=lambda v: v[0])
    s += "\n".join([v[1] for v in ss])
    print(s)