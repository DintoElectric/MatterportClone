#!/usr/bin/env python3
"""
photos_to_3d.py  -  turn 360-degree photos into a 3D space for the Dinto Space Viewer.

Input : a folder of equirectangular (2:1) JPG/PNG panoramas, for example exported from the Insta360 app.
Output: in the output folder
          panorama-placements.json   where each photo was taken and which way it faces
                                     (open the viewer, start a photo space, Add photos, Import placements)
          cloud.xyz                  a colored point cloud in meters, Z up (the viewer loads it as a point cloud)
          report.txt                 what worked, what did not, and how the scale was chosen

How it works
  1. COLMAP's panorama pipeline (through pycolmap) finds the position and facing of every photo.
  2. The photos are leveled, given a real-world scale, and put in a frame whose origin is the first photo.
  3. A plane sweep between neighboring photos measures depth for every pixel; confident pixels become points.

Install   pip install pycolmap numpy opencv-python-headless
Run       python photos_to_3d.py --photos ./photos --out ./space
Scale     By default the tripod or monopod height (--lens-height, meters) fixes the scale.
          For better accuracy give one known distance: --known-distance IMG_001.jpg IMG_004.jpg 7.62

Limits    Photos need distinctive surfaces and overlap. Large plain walls, repeating patterns, glass, mirrors
          and moving people reduce accuracy or leave gaps. Check report.txt before trusting the result.
"""
import argparse
import json
import math
import pathlib
import shutil
import sys
import tempfile
import time

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit('This tool needs OpenCV:  pip install opencv-python-headless')
try:
    import pycolmap
    from pycolmap.panorama import Matcher, PanoRenderType, PanoramaReconstructionOptions, reconstruct
except ImportError:
    sys.exit('This tool needs pycolmap 3.13 or newer:  pip install --upgrade pycolmap')

IMAGE_EXT = {'.jpg', '.jpeg', '.png'}


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------------------------
def equirect_rays(w, h):
    """Unit ray for every pixel of an equirectangular image, in COLMAP camera axes (x right, y down, z forward)."""
    u = (np.arange(w) + 0.5) / w
    v = (np.arange(h) + 0.5) / h
    lon = (u - 0.5) * 2 * np.pi
    lat = (0.5 - v) * np.pi
    LON, LAT = np.meshgrid(lon, lat)
    return np.stack([np.cos(LAT) * np.sin(LON), -np.sin(LAT), np.cos(LAT) * np.cos(LON)], -1).astype(np.float64)


def rays_to_equirect(p, w, h):
    """Pixel coordinates (float32 maps) of camera-frame directions p (..., 3)."""
    n = np.linalg.norm(p, axis=-1)
    lon = np.arctan2(p[..., 0], p[..., 2])
    lat = np.arcsin(np.clip(-p[..., 1] / np.maximum(n, 1e-12), -1, 1))
    return ((lon / (2 * np.pi) + 0.5) * w - 0.5).astype(np.float32), ((0.5 - lat / np.pi) * h - 0.5).astype(np.float32)


def rotation_between(a, b):
    """Rotation matrix that turns unit vector a into unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * (1.0 / (1.0 + c))


# ----------------------------------------------------------------------------------------------
# step 1: structure from motion
# ----------------------------------------------------------------------------------------------
def run_sfm(photos, work, sfm_width, matcher, use_gpu, threads):
    sfm_in = work / 'sfm_in'
    sfm_in.mkdir(parents=True, exist_ok=True)
    for p in photos:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f'Could not read {p.name}')
        h, w = img.shape[:2]
        if abs(w / h - 2) > 0.1:
            log(f'  warning: {p.name} is {w}x{h}, not 2:1. It will probably not reconstruct. Export it as equirectangular.')
        if w > sfm_width:
            img = cv2.resize(img, (sfm_width, sfm_width // 2), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(sfm_in / p.name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    out = work / 'colmap'
    out.mkdir(exist_ok=True)
    t0 = time.time()
    recs = reconstruct(sfm_in, out, PanoramaReconstructionOptions(
        matcher=Matcher(matcher), render_type=PanoRenderType.SPHERICAL, use_gpu=use_gpu, num_threads=threads, show_progress=False))
    log(f'  structure from motion took {time.time() - t0:.0f} s')
    if not recs:
        raise RuntimeError('COLMAP could not reconstruct anything. The photos probably do not overlap enough or look too similar.')
    best = max(recs.values(), key=lambda r: r.num_reg_images())
    return best


# ----------------------------------------------------------------------------------------------
# step 2: level, scale and place
# ----------------------------------------------------------------------------------------------
def estimate_floor(zs, cam_z):
    """Height of the floor from the density of points below the cameras (units of the reconstruction)."""
    below = zs[zs < cam_z - 1e-6]
    if len(below) < 50:
        return None
    lo, hi = np.percentile(below, 2), cam_z
    hist, edges = np.histogram(below, bins=120, range=(lo, hi))
    lowest = hist[: len(hist) // 2]
    k = int(np.argmax(lowest))
    sel = below[(below >= edges[max(k - 1, 0)]) & (below <= edges[min(k + 2, len(edges) - 1)])]
    return float(np.median(sel))


def place(rec, names_order, lens, known, report):
    ims = {im.name: im for im in rec.images.values()}
    names = [n for n in names_order if n in ims]
    C = np.array([ims[n].projection_center() for n in names], dtype=np.float64).reshape(-1, 3)
    Rcw = [ims[n].cam_from_world().rotation.matrix() for n in names]
    fwd = np.array([R.T @ np.array([0, 0, 1.0]) for R in Rcw])        # image center direction
    up = np.array([R.T @ np.array([0, -1.0, 0]) for R in Rcw])        # photo "up"
    up_mean = up.mean(0)
    up_mean /= np.linalg.norm(up_mean)
    tilt = np.degrees(np.arccos(np.clip(up @ up_mean, -1, 1)))
    report.append(f'Photo tilt between photos: largest {tilt.max():.1f} degrees' + ('  (check the camera was level)' if tilt.max() > 3 else ''))
    Rl = rotation_between(up_mean, np.array([0, 0, 1.0]))
    C = (Rl @ C.T).T
    fwd = (Rl @ fwd.T).T
    # first photo faces +Y
    f0 = fwd[0] * np.array([1, 1, 0])
    yaw0 = math.atan2(f0[1], f0[0])
    th = math.pi / 2 - yaw0
    Rz = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]])
    Rg = Rz @ Rl
    C = (Rz @ C.T).T
    fwd = (Rz @ fwd.T).T
    pts = np.array([p.xyz for p in rec.points3D.values()], dtype=np.float64).reshape(-1, 3)
    cols = np.array([p.color for p in rec.points3D.values()], dtype=np.uint8).reshape(-1, 3)
    pts = (Rg @ pts.T).T
    cam_z = float(C[:, 2].mean())
    floor = estimate_floor(pts[:, 2], cam_z)
    # scale
    if known:
        a, b, meters = known
        if a not in names or b not in names:
            raise RuntimeError(f'--known-distance names must be registered photos. Registered: {", ".join(names)}')
        d = np.linalg.norm(C[names.index(a)] - C[names.index(b)])
        scale = meters / d
        report.append(f'Scale: from the known distance {a} to {b} = {meters} m.')
    elif floor is not None:
        scale = lens / (cam_z - floor)
        report.append(f'Scale: from the lens height ({lens} m) above the detected floor. '
                      'This is only as good as that height. A known distance is more accurate (--known-distance).')
    else:
        scale = 1.0
        report.append('Scale: NOT SET. No floor was found. Use --known-distance A B METERS or the viewer\'s Set scale tool.')
    if floor is None:
        floor = cam_z - lens / scale
        report.append('Floor: not detected, assumed at the lens height below the cameras.')
    origin = np.array([C[0, 0], C[0, 1], floor])
    C = (C - origin) * scale
    pts = (pts - origin) * scale
    placements = []
    for i, n in enumerate(names):
        fx, fy = fwd[i, 0], fwd[i, 1]
        heading = math.degrees(math.atan2(-fx, fy))
        placements.append({'name': n, 'x': round(float(C[i, 0]), 4), 'y': round(float(C[i, 1]), 4),
                           'z': round(float(C[i, 2] - lens), 4), 'heading': round(heading, 3), 'lens': lens})
    return placements, pts, cols, Rg, origin, scale


# ----------------------------------------------------------------------------------------------
# step 3: dense depth by plane sweep between neighboring photos
# ----------------------------------------------------------------------------------------------
def depth_map(ref, names, C, R, gray, rays_cam, depths, k_neighbors, patch, min_ncc, min_contrast, lat):
    """Plane sweep: for each candidate depth, warp the neighboring photos onto this one and score how well they match."""
    H, W = gray[ref].shape
    steps = len(depths)
    dist = sorted((np.linalg.norm(C[n] - C[ref]), n) for n in names if n != ref)
    nbrs = [n for _, n in dist[:k_neighbors]]
    rays_w = rays_cam @ R[ref]
    ref_g = gray[ref]
    mean_r = cv2.boxFilter(ref_g, -1, (patch, patch))
    var_r = np.maximum(cv2.boxFilter(ref_g * ref_g, -1, (patch, patch)) - mean_r ** 2, 1e-6)
    cost = np.empty((steps, H, W), np.float32)
    for di, d in enumerate(depths):
        X = C[ref] + d * rays_w
        per = []
        for n in nbrs:
            pc = (X - C[n]) @ R[n].T
            mx, my = rays_to_equirect(pc, W, H)
            warped = cv2.remap(gray[n], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            mean_w = cv2.boxFilter(warped, -1, (patch, patch))
            var_w = np.maximum(cv2.boxFilter(warped * warped, -1, (patch, patch)) - mean_w ** 2, 1e-6)
            cov = cv2.boxFilter(ref_g * warped, -1, (patch, patch)) - mean_r * mean_w
            per.append(1.0 - np.clip(cov / np.sqrt(var_r * var_w), -1, 1))
        per = np.stack(per)
        if len(nbrs) >= 3:
            per = np.partition(per, 1, axis=0)[:2]           # best two views, so one occluded view does not spoil it
        cost[di] = per.mean(0)
    best = np.argmin(cost, axis=0)
    cmin = np.take_along_axis(cost, best[None], 0)[0]
    contrast = cost.mean(0) - cmin
    bi = np.clip(best, 1, steps - 2)
    c0 = np.take_along_axis(cost, (bi - 1)[None], 0)[0]
    c1 = np.take_along_axis(cost, bi[None], 0)[0]
    c2 = np.take_along_axis(cost, (bi + 1)[None], 0)[0]
    denom = c0 - 2 * c1 + c2
    with np.errstate(divide='ignore', invalid='ignore'):
        off = np.where(np.abs(denom) > 1e-9, 0.5 * (c0 - c2) / denom, 0.0).clip(-1, 1)
    depth = np.exp(np.interp(bi + off, np.arange(steps), np.log(depths))).astype(np.float32)
    ok = (cmin < (1 - min_ncc)) & (contrast > min_contrast) & (best > 0) & (best < steps - 1) & (lat < 75)
    med = cv2.medianBlur(depth, 5)
    ok &= np.abs(depth - med) < 0.04 * med
    return depth, ok, cmin, contrast, rays_w


def dense_points(photos_by_name, placements, rec, Rg, origin, scale, width, steps, near, far, k_neighbors, patch, min_ncc, min_contrast, min_consistent, report):
    ims = {im.name: im for im in rec.images.values()}
    names = [p['name'] for p in placements]
    W, H = width, width // 2
    rays_cam = equirect_rays(W, H)
    gray, color, C, R = {}, {}, {}, {}
    for n in names:
        img = cv2.imread(str(photos_by_name[n]), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        color[n] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray[n] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        R[n] = ims[n].cam_from_world().rotation.matrix()
        C[n] = np.asarray(ims[n].projection_center(), dtype=np.float64).reshape(3)
    depths = np.geomspace(near, far, steps) / scale          # candidate depths, in reconstruction units
    lat = np.abs(np.linspace(90, -90, H))[:, None]
    maps, raysw = {}, {}
    # pass 1: a depth map for every photo
    for ref in names:
        t0 = time.time()
        depth, ok, cmin, contrast, rays_w = depth_map(ref, names, C, R, gray, rays_cam, depths, k_neighbors, patch, min_ncc, min_contrast, lat)
        raysw[ref] = rays_w
        maps[ref] = (depth, ok)
        log(f'  depth map {ref}: {100 * ok.mean():.0f}% of pixels passed the photo test ({time.time() - t0:.0f} s)')
    # pass 2: keep only points that another photo's depth map agrees with
    all_pts, all_cols = [], []
    for ref in names:
        depth, ok = maps[ref]
        X = C[ref] + depth[..., None].astype(np.float64) * raysw[ref]
        agree = np.zeros((H, W), np.int32)
        others = sorted((np.linalg.norm(C[n] - C[ref]), n) for n in names if n != ref)[:max(k_neighbors, 2)]
        for _, n in others:
            pc = (X - C[n]) @ R[n].T
            mx, my = rays_to_equirect(pc, W, H)
            dn = cv2.remap(maps[n][0], mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
            okn = cv2.remap(maps[n][1].astype(np.uint8), mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP) > 0
            expect = np.linalg.norm(pc, axis=-1)
            agree += (okn & (np.abs(dn - expect) < 0.03 * expect)).astype(np.int32)
        keep = ok & (agree >= min_consistent)
        pts = ((Rg @ X[keep].T).T - origin) * scale
        all_pts.append(pts)
        all_cols.append(color[ref][keep])
        report.append(f'Dense: {ref} kept {int(keep.sum()):,} points ({100 * keep.mean():.0f}% of pixels; {100 * keep.sum() / max(ok.sum(), 1):.0f}% of those that passed the photo test)')
    if not all_pts:
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    return np.concatenate(all_pts), np.concatenate(all_cols)


def voxel_thin(pts, cols, size):
    if len(pts) == 0:
        return pts, cols
    q = np.floor(pts / size).astype(np.int64)
    q -= q.min(0)
    key = q[:, 0] + (q[:, 1] << 21) + (q[:, 2] << 42)
    _, idx = np.unique(key, return_index=True)
    return pts[idx], cols[idx]


def density_filter(pts, cols, report, cell=0.10, min_count=6):
    """Drop isolated points. Real surfaces have many points in every 10 cm cube, stray matches do not."""
    if len(pts) == 0:
        return pts, cols
    q = np.floor(pts / cell).astype(np.int64)
    q -= q.min(0)
    key = q[:, 0] + (q[:, 1] << 21) + (q[:, 2] << 42)
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    keep = cnt[inv] >= min_count
    report.append(f'Density filter: removed {int((~keep).sum()):,} isolated points ({100 * (~keep).mean():.1f}%).')
    return pts[keep], cols[keep]


def drop_outliers(pts, cols, report, max_range):
    """Remove points far from the main body of the scene."""
    if len(pts) == 0:
        return pts, cols
    r = np.linalg.norm(pts - np.median(pts, 0), axis=1)
    keep = r < max_range
    lo, hi = np.percentile(pts, 0.2, axis=0), np.percentile(pts, 99.8, axis=0)
    pad = 0.15 * (hi - lo) + 0.3
    keep &= np.all((pts >= lo - pad) & (pts <= hi + pad), axis=1)
    report.append(f'Outlier filter: removed {int((~keep).sum()):,} points far from the main body of the scene.')
    return pts[keep], cols[keep]


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--photos', required=True, type=pathlib.Path, help='folder of equirectangular JPG or PNG photos')
    ap.add_argument('--out', required=True, type=pathlib.Path, help='output folder')
    ap.add_argument('--lens-height', type=float, default=1.5, help='camera height above the floor in meters (default 1.5)')
    ap.add_argument('--known-distance', nargs=3, metavar=('PHOTO_A', 'PHOTO_B', 'METERS'), help='real distance between two photo positions')
    ap.add_argument('--matcher', default='exhaustive', choices=['exhaustive', 'sequential', 'vocabtree', 'spatial'], help='exhaustive suits up to about 100 photos')
    ap.add_argument('--sfm-width', type=int, default=3072, help='width photos are shrunk to for the pose step')
    ap.add_argument('--no-dense', action='store_true', help='only find positions, skip the point cloud')
    ap.add_argument('--dense-width', type=int, default=1024, help='width used for depth (1024 = fast, 2048 = detailed and slow)')
    ap.add_argument('--depth-steps', type=int, default=128)
    ap.add_argument('--near', type=float, default=0.5, help='nearest depth in meters')
    ap.add_argument('--far', type=float, default=25.0, help='farthest depth in meters')
    ap.add_argument('--neighbors', type=int, default=4, help='how many nearby photos each photo is compared with')
    ap.add_argument('--min-ncc', type=float, default=0.80, help='how well patches must agree (0 to 1)')
    ap.add_argument('--min-consistent', type=int, default=1, help='how many other photos must agree with a point (default 1)')
    ap.add_argument('--voxel', type=float, default=0.02, help='thin the cloud to one point per cube of this size, in meters')
    ap.add_argument('--gpu', action='store_true', help='use the GPU for feature extraction (needs a CUDA build)')
    ap.add_argument('--threads', type=int, default=-1)
    ap.add_argument('--keep-work', action='store_true', help='keep the COLMAP working folder')
    a = ap.parse_args()

    photos = sorted(p for p in a.photos.iterdir() if p.suffix.lower() in IMAGE_EXT)
    if len(photos) < 2:
        sys.exit('Need at least two photos in --photos.')
    if any(p.suffix.lower() == '.insp' for p in a.photos.iterdir()):
        log('Note: .insp files are ignored. Export them as 2:1 JPG from the Insta360 app first.')
    a.out.mkdir(parents=True, exist_ok=True)
    work = pathlib.Path(tempfile.mkdtemp(prefix='photos_to_3d_', dir=a.out))
    report = [f'Photos found: {len(photos)}']
    known = None
    if a.known_distance:
        known = (a.known_distance[0], a.known_distance[1], float(a.known_distance[2]))
    try:
        log(f'1/3  Finding where each of the {len(photos)} photos was taken')
        rec = run_sfm(photos, work, a.sfm_width, a.matcher, a.gpu, a.threads)
        registered = {im.name for im in rec.images.values()}
        missing = [p.name for p in photos if p.name not in registered]
        report.append(f'Photos placed: {len(registered)} of {len(photos)}')
        if missing:
            report.append('NOT placed (too little overlap or too few distinctive features): ' + ', '.join(missing))
        report.append(f'Reconstruction: {rec.num_points3D():,} sparse points, mean reprojection error {rec.compute_mean_reprojection_error():.2f} px')
        log('2/3  Leveling, scaling, naming positions')
        placements, spts, scols, Rg, origin, scale = place(rec, [p.name for p in photos], a.lens_height, known, report)
        by_name = {p.name: p for p in photos}
        doc = {'app': 'dinto-space-viewer', 'version': 1, 'space': a.photos.name, 'panos': placements, 'plan': None}
        (a.out / 'panorama-placements.json').write_text(json.dumps(doc, indent=2))
        pts, cols = spts, scols
        if not a.no_dense and len(placements) >= 2:
            log('3/3  Measuring depth between neighboring photos')
            dpts, dcols = dense_points(by_name, placements, rec, Rg, origin, scale, a.dense_width, a.depth_steps, a.near, a.far,
                                       min(a.neighbors, len(placements) - 1), 7, a.min_ncc, 0.25, a.min_consistent, report)
            if len(dpts):
                pts, cols = dpts, dcols
        pts, cols = drop_outliers(pts, cols, report, 60.0)
        if not a.no_dense:
            pts, cols = density_filter(pts, cols, report)
            pts, cols = drop_outliers(pts, cols, report, 60.0)
        pts, cols = voxel_thin(pts, cols, a.voxel)
        with open(a.out / 'cloud.xyz', 'w') as f:
            np.savetxt(f, np.column_stack([pts, cols]), fmt='%.4f %.4f %.4f %d %d %d')
        report.append(f'Point cloud written: {len(pts):,} points ({"dense" if not a.no_dense else "sparse"}).')
        extent = pts.max(0) - pts.min(0) if len(pts) else np.zeros(3)
        report.append(f'Extent of the cloud: {extent[0]:.1f} x {extent[1]:.1f} x {extent[2]:.1f} m')
    finally:
        if not a.keep_work:
            shutil.rmtree(work, ignore_errors=True)
    text = '\n'.join(report)
    (a.out / 'report.txt').write_text(text + '\n')
    log('\n' + text)
    log(f'\nDone. Files are in {a.out}')


if __name__ == '__main__':
    main()
