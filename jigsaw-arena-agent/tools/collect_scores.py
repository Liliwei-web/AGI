import glob, re, os, csv, datetime
ARENA_LOGS = os.environ.get("ARENA_LOGS", "arena_offline/logs")
logs = glob.glob(os.path.join(ARENA_LOGS, "arena_*.log"))
f = "%Y-%m-%d %H:%M:%S.%f"
rows = []
for p in logs:
    try:
        t = open(p, encoding="utf-8", errors="ignore").read()
    except Exception:
        continue
    m1 = re.search(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+).*Notifying agents to start answering", t, re.M)
    m2 = re.search(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+).*agent_start_evaluation called for agent_id: (\w+)", t, re.M)
    m3 = re.search(r'"jigsaw_score":\s*([\d.]+).*?"score":\s*([\d.]+)', t, re.S)
    if not (m1 and m2):
        continue
    dt = (datetime.datetime.strptime(m2.group(1), f) - datetime.datetime.strptime(m1.group(1), f)).total_seconds()
    js = float(m3.group(1)) if m3 else None
    sc = float(m3.group(2)) if m3 else None
    rows.append((datetime.datetime.strptime(m1.group(1), f).strftime("%Y-%m-%d %H:%M"), m2.group(2), round(dt, 1), js, sc))
rows.sort()
out = os.environ.get("SCORES_CSV", "results/scores.csv")
with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["run_at", "agent_id", "elapsed_sec", "jigsaw_score", "total_score"])
    for r in rows:
        w.writerow(r)
print("rows:", len(rows))
for r in rows:
    print("%s  %s  %6.1fs  jigsaw=%-6s total=%s" % (r[0], r[1], r[2], r[3], r[4]))