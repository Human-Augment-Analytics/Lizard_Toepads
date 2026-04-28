def get_tps_coords(img_id, img, tps_data_dir):
    cls = ["finger", "toe"]
    ret = {}
    h, w = img.shape[:2]
    for c in cls:
        fp = f"{tps_data_dir}/{img_id}_{c}.TPS"
        coordinates = []
        skip = 2
        with open(fp, "r") as f:
            for line in f:
                line = line.strip()
                if not line or "=" in line:
                    continue
                parts = line.split()
                if len(parts) == 2:
                    if skip > 0:
                        skip -= 1
                        continue
                    try:
                        x, y = map(float, parts)
                        coordinates.append((x, h - 1 - y))
                    except ValueError:
                        continue
        ret[c] = coordinates
    return ret
