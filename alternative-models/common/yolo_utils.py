import cv2

def crop_toe_boxes(r, image, g_coords, show=False, output_name="output"):
    target_classes = [2, 3]
    test_classes = [0, 1]
    classmap = {2: "finger", 3: "toe"}
    crops = []
    coords_list = []
    tps = []
    result = r[0]
    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    for (x1, y1, x2, y2), cls_id in zip(xyxy, cls_ids):
        if int(cls_id) in target_classes:
            x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
            crop = image[y1i:y2i, x1i:x2i].copy()
            coords_list.append([x1i, y1i, x2i, y2i])
            l_coords = []
            valid = True
            for (x, y) in g_coords[classmap[int(cls_id)]]:
                x_local = x - x1
                y_local = y - y1
                if not (0 <= x_local < (x2i - x1i) and 0 <= y_local < (y2i - y1i)):
                    valid = False
                    break
                l_coords.append((x_local, y_local))
            if not valid:
                print(f"Warning: skipping crop, landmark outside bounds")
                continue
            crops.append(crop)
            tps.append(l_coords)
            if show:
                copy = crop.copy()
                for (lx, ly) in l_coords:
                    cv2.circle(copy, (int(round(lx)), int(round(ly))), 5, (0, 0, 255), -1)
                cv2.imshow(f"Crop Class {int(cls_id)}", copy)
                cv2.waitKey(0)
                cv2.destroyWindow(f"Crop Class {int(cls_id)}")
        elif int(cls_id) in test_classes:
            x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
            crop = image[y1i:y2i, x1i:x2i].copy()
            outpath = f"{output_name}_{classmap[int(cls_id)+2]}.jpg"
            cv2.imwrite(outpath, crop)
    return crops, coords_list, tps
