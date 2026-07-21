import numpy as np


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


def get_ruler_distance(img_id, img, tps_data_dir):
    """Return the pixel distance between the two ruler landmarks for each class.

    These are the first two coordinate pairs in each TPS file (the ones
    skipped by get_tps_coords). The distance can be used to convert pixel
    error to mm given a known physical ruler length.
    """
    cls = ["finger", "toe"]
    ret = {}
    h, w = img.shape[:2]
    for c in cls:
        fp = f"{tps_data_dir}/{img_id}_{c}.TPS"
        ruler_pts = []
        try:
            with open(fp, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or "=" in line:
                        continue
                    parts = line.split()
                    if len(parts) == 2:
                        try:
                            x, y = map(float, parts)
                            ruler_pts.append(np.array([x, h - 1 - y]))
                        except ValueError:
                            continue
                        if len(ruler_pts) == 2:
                            break
        except FileNotFoundError:
            ret[c] = None
            continue
        if len(ruler_pts) == 2:
            ret[c] = float(np.linalg.norm(ruler_pts[0] - ruler_pts[1]))
        else:
            ret[c] = None
    return ret
