#!/usr/bin/env python3
"""Stage 2 — render the 5 s vertical clip.

A tiny 2.5D puppet rig: the master pose is split into two character sprites that are
composited over a static background plate.  Everything is animated with keyframed
affine moves (hop / squash-stretch / weight shift) plus local, joint-pivoted warps
for secondary motion (ears, scarf, paw, fins).  The camera is a smooth, low amplitude
sway that settles into a locked final frame.

    python tools/2_render.py            # encode out/shark_bunny_dance_5s_9x16.mp4
    python tools/2_render.py --debug    # dump a few preview stills to build/
"""
import numpy as np, cv2, sys, subprocess, types, os
from PIL import Image
import imageio_ffmpeg

W, H = 768, 1376            # native render size (master pose resolution)
FPS, DUR = 30, 5.0
N = int(round(FPS * DUR))
OUT_MP4 = "out/shark_bunny_dance_5s_9x16.mp4"

plate = np.array(Image.open("assets/derived/plate_aligned.png").convert("RGB")).astype(np.float32) / 255.
SH = np.array(Image.open("assets/derived/shark.png").convert("RGBA")).astype(np.float32) / 255.
BU = np.array(Image.open("assets/derived/bunny.png").convert("RGBA")).astype(np.float32) / 255.
GX, GY = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

def cl(x, a=0.0, b=1.0): return np.clip(x, a, b)
def smoothstep(u):
    u = np.clip(u, 0, 1); return u * u * (3 - 2 * u)
def gauss(cx, cy, sig): return np.exp(-(((GX - cx) ** 2 + (GY - cy) ** 2) / (2 * sig * sig))).astype(np.float32)

a_sh, a_bu = SH[..., 3], BU[..., 3]
ear_w = (cl((802.0 - GY) / 150.0) ** 1.5).astype(np.float32) * a_bu
paw_w = (gauss(636, 966, 46) * a_bu).astype(np.float32)
R, G, B = BU[..., 0], BU[..., 1], BU[..., 2]
scar = ((R > 0.62) & (G > 0.32) & (G < 0.74) & (B < 0.36) & (R - B > 0.26)).astype(np.float32)
scarf_w = cl(cv2.GaussianBlur(scar, (0, 0), 8.0) * a_bu / 0.40)
finL_w = (gauss(150, 1000, 44) * a_sh).astype(np.float32)
finR_w = (gauss(382, 988, 40) * a_sh).astype(np.float32)
body_swim = (cl((1150.0 - GY) / 400.0) ** 1.2 * a_sh).astype(np.float32)

def rot_field(w, cx, cy, deg):
    """per-pixel rotation of `deg` (scaled by weight w) around a joint at (cx, cy)"""
    th = np.deg2rad(deg) * w
    c, s = np.cos(th), np.sin(th)
    x = GX - cx; y = GY - cy
    return (c - 1) * x - s * y, s * x + (c - 1) * y

def env_dance(t):
    """rhythm envelope: full 0-3 s, eases down into the slow wave, tiny breath at the end"""
    if t < 3.0: return 1.0
    if t < 3.35: return 1.0 - 0.62 * smoothstep((t - 3.0) / 0.35)
    if t < 4.05: return 0.38 - 0.26 * smoothstep((t - 3.35) / 0.70)
    return 0.12 - 0.08 * smoothstep((t - 4.05) / 0.55)

def wave_lift(t):
    """0 -> 1 while the paw / fins are raised (3.0 - 4.2 s)"""
    if t < 3.00: return 0.0
    L = smoothstep((t - 3.00) / 0.30)
    if t > 3.92: L *= 1.0 - smoothstep((t - 3.92) / 0.28)
    return L

def pose(t):
    ph = 2 * np.pi * 1.20 * t          # 1.2 Hz beat
    e = env_dance(t)
    L = wave_lift(t)
    sweep = np.sin(2 * np.pi * (t - 3.30) / 0.66) if 3.30 < t < 3.96 else 0.0
    hp = 0.5 - 0.5 * np.cos(ph)        # 0 on the ground, 1 at the top of the hop
    p = types.SimpleNamespace()
    p.bu_hop = -34.0 * hp * e
    p.bu_sy = 1.0 - 0.055 * np.cos(ph) * e
    p.bu_sx = 1.0 + 0.045 * np.cos(ph) * e
    p.bu_rot = 3.2 * np.sin(ph + 0.55) * e
    p.sh_hop = -11.0 * (0.5 - 0.5 * np.cos(ph + np.pi)) * e
    p.sh_rot = 4.0 * np.sin(ph + 2.1) * e
    p.sh_sx = 1.0 + 0.020 * np.cos(ph + np.pi) * e
    p.sh_sy = 1.0 - 0.022 * np.cos(ph + np.pi) * e
    p.sh_dx = 5.0 * np.sin(ph + 0.9) * e
    p.ear_ang = 6.5 * np.sin(ph - 1.05) * e + 2.5 * np.sin(ph - 1.05) * L
    p.scarf_dx = 6.0 * np.sin(ph - 1.75) * e
    p.scarf_dy = 2.0 * np.sin(2 * ph - 1.95) * e
    p.paw_ang = 9.0 * L + 21.0 * sweep * L
    p.paw_dx = 13.0 * sweep * L
    p.paw_dy = -5.0 * L
    p.finL_ang = 12.0 * L + 18.0 * sweep * L
    p.finR_ang = -10.0 * L - 15.0 * sweep * L
    p.swim_dy = 4.0 * np.sin(ph + 1.0) * e
    return p

def warp_rgba(rgba, dx, dy):
    mx = (GX + dx).astype(np.float32); my = (GY + dy).astype(np.float32)
    rgb = cv2.remap(rgba[..., :3], mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    a = cv2.remap(rgba[..., 3], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return np.dstack([rgb, np.clip(a, 0, 1)])

def place(rgba, pivot, rot, scale, tx, ty):
    th = np.deg2rad(rot)
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], np.float32)
    A = (Rm @ np.diag(np.array(scale, np.float32))).astype(np.float32)
    pv = np.array(pivot, np.float32)
    t = pv - A @ pv + np.array([tx, ty], np.float32)
    M = np.hstack([A, t.reshape(2, 1)]).astype(np.float32)
    rgb = cv2.warpAffine(rgba[..., :3], M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    a = cv2.warpAffine(rgba[..., 3], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return np.dstack([rgb, np.clip(a, 0, 1)])

def shadow_map(cx, cy, rx, ry, amt):
    m = np.zeros((H, W), np.float32)
    cv2.ellipse(m, (int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360, 1.0, -1)
    return (cv2.GaussianBlur(m, (0, 0), 18) * amt)[..., None]

def render(t):
    p = pose(t)
    frame = plate.copy()
    # contact shadows stay on the ground and soften as each character lifts off
    hb = cl(-p.bu_hop / 34.0); hs = cl(-p.sh_hop / 11.0)
    shd = np.clip(shadow_map(524, 1150, 96 * (1 - 0.22 * hb), 21 * (1 - 0.25 * hb), 0.17 * (1 - 0.45 * hb)) +
                  shadow_map(252, 1152, 118 * (1 - 0.18 * hs), 23 * (1 - 0.20 * hs), 0.16 * (1 - 0.40 * hs)), 0, 1)
    frame = frame * (1 - shd * 0.9)
    # shark
    dxf, dyf = rot_field(finL_w, 190, 985, p.finL_ang)
    dxf2, dyf2 = rot_field(finR_w, 355, 975, p.finR_ang)
    s = warp_rgba(SH, dxf + dxf2, dyf + dyf2 + body_swim * p.swim_dy)
    s = place(s, (250.0, 1130.0), p.sh_rot, (p.sh_sx, p.sh_sy), p.sh_dx, p.sh_hop)
    frame = frame * (1 - s[..., 3:4]) + s[..., :3] * s[..., 3:4]
    # bunny
    ex, ey = rot_field(ear_w, 525, 802, p.ear_ang)
    px_, py_ = rot_field(paw_w, 605, 985, p.paw_ang)
    b = warp_rgba(BU, ex + px_ + scarf_w * p.scarf_dx + paw_w * p.paw_dx,
                  ey + py_ + scarf_w * p.scarf_dy + paw_w * p.paw_dy)
    b = place(b, (525.0, 1132.0), p.bu_rot, (p.bu_sx, p.bu_sy), 0.0, p.bu_hop)
    frame = frame * (1 - b[..., 3:4]) + b[..., :3] * b[..., 3:4]
    return frame

# ---- camera: soft rhythmic sway, then a short pull back that locks ----
T_CAM_END, T_LOCK = 3.0, 4.62
def zoom(t):
    base = 1.042 + 0.006 * np.sin(2 * np.pi * 1.20 * t - 0.4)
    if t < T_CAM_END: return base
    if t < T_LOCK:
        u = smoothstep((t - T_CAM_END) / (T_LOCK - T_CAM_END))
        z0 = 1.042 + 0.006 * np.sin(2 * np.pi * 1.20 * T_CAM_END - 0.4)
        return z0 + (1.0 - z0) * u
    return 1.0

def cam_off(t):
    if t >= T_LOCK: return 0.0, 0.0, 0.0
    fade = 1.0 if t < T_CAM_END else 1.0 - 0.55 * smoothstep((t - T_CAM_END) / (T_LOCK - T_CAM_END))
    return (2.4 * np.sin(2 * np.pi * 0.60 * t + 0.9) * fade,
            2.8 * np.sin(2 * np.pi * 1.20 * t + 2.0) * fade,
            0.26 * np.sin(2 * np.pi * 0.60 * t - 0.5) * fade)

def camera(img, t):
    z = zoom(t); tx, ty, rot = cam_off(t)
    M = cv2.getRotationMatrix2D(((W - 1) / 2, (H - 1) / 2), rot, z)
    M[0, 2] += tx; M[1, 2] += ty
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

# ---- grade: pastel lift, gentle bloom, warm glow ramp over the last second ----
def bloom(f, strength, thr=0.70, sig=20):
    m = cl((f.max(axis=2) - thr) / (1 - thr)) ** 1.4
    bl = cv2.GaussianBlur(f * m[..., None], (0, 0), sig)
    return 1 - (1 - f) * (1 - cl(bl * strength))

def warm_glow(f, amt):
    h, w = f.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - w * 0.52) / (w * 0.80)) ** 2 + ((yy - h * 0.10) / (h * 1.00)) ** 2)
    g = np.exp(-(d * d) * 1.7)[..., None]
    return 1 - (1 - f) * (1 - cl(g * np.array([1.0, 0.85, 0.60], np.float32) * amt))

def grade(f, t):
    f = cl(f) * 0.93 + 0.052
    l = f.mean(axis=2, keepdims=True)
    f = cl(l + (f - l) * 1.07)
    f = bloom(f, 0.15)
    amt = 0.115 * smoothstep((t - 4.0) / 0.95) if t > 4.0 else 0.0
    return cl(warm_glow(f, amt))

def frame_at(t):
    f = camera(render(t), t)
    f = cv2.resize(f, (1080, 1935), interpolation=cv2.INTER_LANCZOS4)[7:7 + 1920]
    return (grade(f, t) * 255.0 + 0.5).astype(np.uint8)

if __name__ == "__main__":
    os.makedirs("out", exist_ok=True)
    os.makedirs("build", exist_ok=True)
    if "--debug" in sys.argv:
        for t in [0.0, 0.42, 1.0, 2.5, 3.2, 3.5, 3.75, 4.4, 4.95]:
            Image.fromarray(frame_at(t)).resize((540, 960), Image.LANCZOS).save(f"build/preview_{t:.2f}.jpg", quality=92)
        print("previews in build/"); sys.exit(0)
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1080x1920", "-r", str(FPS), "-i", "-",
           "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
           "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "slow", "-crf", "17",
           "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.2", "-g", "30",
           "-movflags", "+faststart", "-c:a", "aac", "-b:a", "128k", "-shortest", "-frames:v", str(N), OUT_MP4]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(N):
        pr.stdin.write(frame_at(i / FPS).tobytes())
        if i % 30 == 0: print("frame", i, flush=True)
    pr.stdin.close(); pr.wait()
    print("exit", pr.returncode, OUT_MP4)
