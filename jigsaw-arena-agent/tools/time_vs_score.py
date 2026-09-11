import glob, re, os, json, datetime
ARENA_LOGS = os.environ.get("ARENA_LOGS", "arena_offline/logs")
logs = glob.glob(os.path.join(ARENA_LOGS, "arena_*.log"))
rows = []
for p in logs:
    try:
        t = open(p, encoding="utf-8", errors="ignore").read()
    except Exception:
        continue
    m1 = re.search(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+).*Notifying agents to start answering", t, re.M)
    m2 = re.search(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+).*agent_start_evaluation", t, re.M)
    m3 = re.search(r'"jigsaw_score":\s*([\d.]+).*?"score":\s*([\d.]+)', t, re.S)
    if not (m1 and m2 and m3):
        continue
    f = "%Y-%m-%d %H:%M:%S.%f"
    dt = (datetime.datetime.strptime(m2.group(1), f) - datetime.datetime.strptime(m1.group(1), f)).total_seconds()
    rows.append((os.path.getmtime(p), dt, float(m3.group(1)), float(m3.group(2))))
rows.sort()
print("  elapsed   jigsaw   total")
for _, dt, js, sc in rows[-16:]:
    print("%8.1fs  %7.1f  %6.2f" % (dt, js, sc))