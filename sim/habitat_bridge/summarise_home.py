import json, glob, sys, os, numpy as np
root = sys.argv[1]
dirs = sorted(set(os.path.relpath(os.path.dirname(f), root) for f in glob.glob(f"{root}/**/brain_day*_ep*.json", recursive=True)))
for c in dirs:
    fs = sorted(glob.glob(f"{root}/{c}/brain_day*_ep*.json"), key=lambda f: int(f.split("day")[1].split("_")[0]))
    if not fs: continue
    rows = []
    for f in fs:
        r = json.load(open(f)); h = r["home"]
        rows.append((r["day"], r["episode"], h["first_bowl_s"], round(h["near_bowl_s"],1), round(h["eating_s"],1), h["mean_bowl_dist"], round(h["at_plant_s"],1), h["pets"], h["treats"], round(h["near_person_s"],1), r["collisions"], r.get("wall_s")))
    print("==", c, "(day, ep, first_bowl_s, near_bowl_s, eating_s, mean_bowl_dist, plant_s, pets, treats, near_person_s, bumps, wall_s)")
    for x in rows: print("  ", x)
    fb = [x[2] for x in rows]
    print("   reached bowl %d/%d, mean near %.1f s, eating %.1f s, bumps %d" % (sum(v is not None for v in fb), len(rows), np.mean([x[3] for x in rows]), np.mean([x[4] for x in rows]), sum(x[10] for x in rows)))
