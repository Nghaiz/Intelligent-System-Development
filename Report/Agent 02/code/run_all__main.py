# scripts/run_all.py
def main():
    t0 = time.perf_counter()
    js = jobs()
    workers = min(24, os.cpu_count() or 4)
    with Pool(workers) as pool:
        res_async = pool.map_async(run_job, js, chunksize=1)
        base = baseline()
        results = res_async.get()
    by_exp = {}
    for r in results:
        by_exp.setdefault(r["exp"], []).append(r)
    for exp, rs in by_exp.items():
        (RAW / f"{exp}.json").write_text(json.dumps(rs, ensure_ascii=False), encoding="utf-8")
    msg = (f"{len(js)} lượt + baseline, {workers} tiến trình, {time.perf_counter() - t0:.1f} s; "
           f"baseline {base['eval']['mean'][0]:.4f} chính sách {base['policy']}")
    (OUT / "logs").mkdir(exist_ok=True)
    (OUT / "logs" / "run_all.log").write_text(msg + "\n", encoding="utf-8")
    print(msg)
