import csv
import sqlite3
import sys
from pathlib import Path

# Dump every kernel launch from Nsight Systems sqlite exports into one CSV.
#   nsys export --type sqlite v100-1234.nsys-rep      (or the .sqlite that `nsys profile --stats=true` leaves behind)
#   python nsys_to_csv.py kernel_launches.csv v1=v100-2135647.sqlite v1.1=v100-2135651.sqlite ...

out = Path(sys.argv[1])
rows = []
for arg in sys.argv[2:]:
    version, path = arg.split("=", 1)
    job = Path(path).stem.split("-")[-1]
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    names = dict(db.execute("select id, value from StringIds"))
    launches = db.execute(
        "select start, end, streamId, shortName, blockX, gridX, registersPerThread, dynamicSharedMemory "
        "from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
    stage_count = {}  # FFT stage index, counted separately for each block size
    for start, end, stream, name_id, block, grid, regs, dyn_smem in launches:
        name = names[name_id]
        stage = ""
        if "stage" in name:
            stage_count[block] = stage_count.get(block, 0) + 1
            stage = stage_count[block]
        rows.append(dict(version=version, job=job, kernel=name, stream=stream, threads_per_block=block,
                         grid=grid, stage=stage, registers=regs, dyn_smem_bytes=dyn_smem,
                         start_ns=start, duration_ms=round((end - start) / 1e6, 4)))

with out.open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"{len(rows)} kernel launches -> {out}")
