#!/usr/bin/env python3
"""Stage 1 — build the render layer set.

Runs optical-free compositing prep:
  1. aligns the empty background plate (assets/plate.png) to the master pose (assets/keyframes/k4.png)
  2. mattes the two characters out of the master pose with a watershed split (blue shark / yellow bunny)
  3. writes assets/derived/{plate_aligned,shark,bunny}.png used by tools/2_render.py
"""
import numpy as np, cv2, os
from PIL import Image

MASTER = "assets/keyframes/k4.png"
PLATE = "assets/plate.png"
OUT = "assets/derived"
os.makedirs(OUT, exist_ok=True)

master = np.array(Image.open(MASTER).convert("RGB"))
plate0 = np.array(Image.open(PLATE).convert("RGB"))
H, W = master.shape[:2]

def gray(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return cv2.GaussianBlur(g, (0, 0), 1.6)

warp = np.eye(2, 3, dtype=np.float32)
cc, warp = cv2.findTransformECC(gray(master), gray(plate0), warp, cv2.MOTION_EUCLIDEAN,
                                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-7), None, 5)
plate = cv2.warpAffine(plate0, warp, (W, H), flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                       borderMode=cv2.BORDER_REFLECT)
Image.fromarray(plate).save(f"{OUT}/plate_aligned.png")
print(f"plate aligned, ecc={cc:.4f}")

pf = cv2.GaussianBlur(plate.astype(np.float32), (0, 0), 1.2)

def fill_holes(m):
    ff = m.copy()
    cv2.floodFill(ff, np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8), (0, 0), 1)
    return np.clip(m | (ff == 0).astype(np.uint8), 0, 1)

def biggest(m, k=1):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n <= 1:
        return m
    idx = np.argsort(st[1:, cv2.CC_STAT_AREA])[::-1][:k] + 1
    out = np.zeros_like(m)
    for i in idx:
        out[lab == i] = 1
    return out

f = cv2.GaussianBlur(master.astype(np.float32), (0, 0), 1.2)
d = np.abs(f - pf).max(axis=2) / 255.0
keep = (np.clip((d - 0.030) / (0.085 - 0.030), 0, 1) > 0.35).astype(np.uint8)
keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
keep = biggest(keep, 2)
keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8), iterations=2)

R, G, B = master[..., 0].astype(np.int32), master[..., 1].astype(np.int32), master[..., 2].astype(np.int32)
blue = ((B > R + 12) & (keep > 0)).astype(np.uint8)
yellow = (((R > B + 18) & (G > B + 6)) & (keep > 0)).astype(np.uint8)
blue = cv2.erode(blue, np.ones((7, 7), np.uint8))
yellow = cv2.erode(yellow, np.ones((7, 7), np.uint8))
markers = np.zeros((H, W), np.int32)
markers[keep == 0] = 1
markers[blue > 0] = 2
markers[yellow > 0] = 3
cv2.watershed(cv2.cvtColor(master, cv2.COLOR_RGB2BGR), markers)

masks = {}
for name, val in (("shark", 2), ("bunny", 3)):
    m = (markers == val).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    m = fill_holes(m)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = fill_holes(biggest(m, 1))
    masks[name] = m

covered = np.clip(masks["shark"] + masks["bunny"], 0, 1)
left = ((keep == 1) & (covered == 0)).astype(np.uint8)
if left.sum() > 0:
    d1 = cv2.distanceTransform(1 - masks["shark"], cv2.DIST_L2, 5)
    d2 = cv2.distanceTransform(1 - masks["bunny"], cv2.DIST_L2, 5)
    masks["shark"] = np.clip(masks["shark"] + (left & (d1 <= d2)), 0, 1)
    masks["bunny"] = np.clip(masks["bunny"] + (left & (d1 > d2)), 0, 1)

for name, m in masks.items():
    grown = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8))
    alpha = np.clip(cv2.GaussianBlur(grown.astype(np.float32), (0, 0), 1.5) * 1.06, 0, 1)
    rgba = np.dstack([master, (alpha * 255).astype(np.uint8)])
    Image.fromarray(rgba).save(f"{OUT}/{name}.png")
    print(name, "sprite written", rgba.shape)
