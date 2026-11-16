"""Search for missing power-line spans using existing lines, poles and image ridges."""
from itertools import combinations

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, uniform_filter1d


def unit(vector):
    return vector / max(np.linalg.norm(vector), 1e-6)


def wire_seeds(paths, limit=36):
    seeds = []
    for points in sorted(paths, key=len, reverse=True):
        points = np.asarray(points, dtype=float)
        chord = np.linalg.norm(points[-1] - points[0])
        if chord < 90:
            continue
        arc = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
        if arc > 1.24 * chord:
            continue
        direction = unit(points[-1] - points[0])
        if direction[0] < 0 or (direction[0] == 0 and direction[1] < 0):
            points, direction = points[::-1], -direction
        seeds.append((points, direction))
        if len(seeds) == limit:
            break
    return seeds


def search_axes(seeds, poles):
    """Axes indicate where to search. They are not themselves predictions."""
    top = []
    for pole in poles:
        x0, y0, x1, y1 = pole['box']
        if pole['score'] < .28:
            continue
        height = min(110, (y1-y0)*.22)
        top.append((pole['id'], np.array([(x0+x1)/2, y0+height/2]),
                    float(np.clip(.5*(x1-x0)+height/2+22, 35, 120))))
    axes = []
    for (a, u), (b, v) in combinations(seeds, 2):
        if abs(u @ v) < .978:
            continue
        if u @ v < 0:
            b, v = b[::-1], -v
        direction = unit(u + v)
        normal = np.array([-direction[1], direction[0]])
        # All calculations use the mean axis, including nearly vertical lines.
        at, bt = a @ direction, b @ direction
        an, bn = np.median(a @ normal), np.median(b @ normal)
        separation = abs(an - bn)
        lo, hi = max(at[0], bt[0]), min(at[-1], bt[-1])
        if hi - lo >= 70 and 10 <= separation <= 140:
            start = max(min(at[0], bt[0]) - 130, lo - 270)
            end = min(max(at[-1], bt[-1]) + 130, hi + 270)
            if end-start > 850:
                mid = (lo+hi)/2
                start, end = max(start, mid-425), min(end, mid+425)
            for fraction in (-1, .5, 2):
                offset = an + fraction * (bn - an)
                radius = min(17, max(6, separation * .27))
                axes.append(dict(kind='parallel', start=direction*start + normal*offset,
                                 end=direction*end + normal*offset, radius=radius,
                                 spacing=float(separation), fraction=fraction))
        elif lo - hi >= 55:
            left, right = (a, b) if at[-1] <= bt[0] else (b, a)
            p, q = left[-1], right[0]
            gap = float(np.linalg.norm(q-p))
            lateral = abs((q-p) @ normal)
            if 60 <= gap <= 700 and lateral <= max(9, .07*gap):
                axes.append(dict(kind='gap', start=p, end=q,
                                 radius=min(26, max(10, gap*.045))))

    for points, direction in seeds:
        for p, tangent in ((points[0], -direction), (points[-1], direction)):
            for ident, center, radius in top:
                delta = center - p
                length = float(delta @ tangent)
                lateral = np.linalg.norm(delta - length*tangent)
                if 85 <= length <= 750 and lateral <= radius:
                    # Project onto the seed's direction rather than aiming at a box center.
                    axes.append(dict(kind='pole', start=p, end=p+length*tangent,
                                     radius=min(25, 10+length*.03), pole=ident))
    for item in axes:
        # Use pole proximity to prioritize a corridor even when a pole is not an endpoint.
        if 'pole' in item:
            continue
        start, end = item['start'], item['end']
        vector = end-start
        for ident, center, radius in top:
            fraction = float((center-start) @ vector / (vector @ vector))
            if -.18 <= fraction <= 1.18 and np.linalg.norm(center-start-fraction*vector) <= radius:
                item['near_pole'] = ident
                break
    return axes


def longest_supported_sections(support, max_gap=22):
    seen = np.flatnonzero(support)
    return [group for group in np.split(seen, np.where(np.diff(seen) > max_gap+1)[0] + 1)
            if len(group)]


def follow_ridge(gray, distance, axis):
    start, end = axis['start'], axis['end']
    direction = unit(end-start)
    normal = np.array([-direction[1], direction[0]])
    length = np.linalg.norm(end-start)
    rows = int(length/4)+1
    radius = round(axis['radius'])
    offsets = np.arange(-radius, radius+1)
    centers = np.linspace(start, end, rows)
    grid = centers[:, None, :] + offsets[None, :, None]*normal

    def sample(image, points, interpolation=cv2.INTER_LINEAR):
        xy = points.astype(np.float32)
        return cv2.remap(image, xy[..., 0], xy[..., 1], interpolation,
                         borderMode=cv2.BORDER_REPLICATE)

    center = sample(gray, grid)
    contrasts = []
    for width in (2, 4, 6):
        contrasts.append((sample(gray, grid-width*normal)-center,
                          sample(gray, grid+width*normal)-center))
    values = np.stack(contrasts)
    novelty = sample(distance, grid, cv2.INTER_NEAREST)
    cols = len(offsets)
    indexes = np.arange(cols)
    moves = np.arange(-2, 3)
    previous = indexes[None, :] + moves[:, None]
    within = (previous >= 0) & (previous < cols)
    previous = np.clip(previous, 0, cols-1)
    in_frame = ((grid[..., 0] >= 7) & (grid[..., 0] < gray.shape[1]-7) &
                (grid[..., 1] >= 7) & (grid[..., 1] < gray.shape[0]-7))
    found = []
    for sign in (1, -1):
        ridge = np.maximum(0, (sign*values).min(axis=1).max(axis=0))
        smoothed = uniform_filter1d(ridge, 5, axis=0, mode='nearest')
        reward = np.minimum(12., .5*ridge + .5*smoothed) - .12*np.abs(offsets)
        reward[~in_frame] = -18
        if axis['kind'] == 'parallel':
            reward[novelty < 6] -= 9
        costs = reward[0].copy()
        if axis['kind'] == 'gap':
            costs -= 1.5*np.abs(offsets)
        back = np.zeros((rows, cols), dtype=np.int16)
        for i in range(1, rows):
            options = np.where(within, costs[previous] - 1.15*np.abs(moves[:, None]), -np.inf)
            best = options.argmax(axis=0)
            costs = options[best, indexes] + reward[i]
            back[i] = previous[best, indexes]
        if axis['kind'] == 'gap':
            costs -= 1.5*np.abs(offsets)
        selected = np.empty(rows, dtype=int)
        selected[-1] = costs.argmax()
        for i in range(rows-1, 0, -1):
            selected[i-1] = back[i, selected[i]]
        i = np.arange(rows)
        strength = ridge[i, selected]
        valid = in_frame[i, selected]
        unseen = novelty[i, selected] >= (6 if axis['kind'] == 'parallel' else 3)
        supported = (strength >= 2.4) & valid
        for section in longest_supported_sections(supported):
            left, right = section[0], section[-1]+1
            window = slice(left, right)
            samples = right-left
            if samples < (25 if axis['kind'] == 'parallel' else 15):
                continue
            coverage = float(np.mean(supported[window]))
            new_fraction = float(np.mean(unseen[window]))
            if coverage < .39 or new_fraction < (.72 if axis['kind']=='parallel' else .43):
                continue
            if not valid[window].all():
                continue
            points = np.rint(grid[i, selected][window]).astype(int)
            strength_mean = float(np.mean(np.minimum(strength[window], 15)))
            score = (coverage * np.sqrt(samples) * (2+strength_mean) *
                     (.7+.3*new_fraction))
            found.append(dict(xy=points.tolist(), kind=axis['kind'],
                              length=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),
                              support=coverage, contrast=strength_mean, novelty=new_fraction,
                              score=float(score), **{k:axis[k] for k in ('pole','near_pole','spacing','fraction') if k in axis}))
    return max(found, key=lambda item: item['score'], default=None)


def similarity(a, b):
    a, b = np.asarray(a)[::2], np.asarray(b)[::2]
    distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return max(np.mean(distances.min(axis=0) <= 7), np.mean(distances.min(axis=1) <= 7))


def generate(photo, old_mask, paths, poles, limit):
    gray = cv2.cvtColor(np.asarray(photo), cv2.COLOR_RGB2GRAY).astype(np.float32)
    distance = (distance_transform_edt(~old_mask).astype(np.float32) if old_mask.any()
                else np.full(old_mask.shape, 1000, dtype=np.float32))
    seeds = wire_seeds(paths)
    axes = search_axes(seeds, poles)
    proposals = []
    ranked_axes = sorted(axes, key=lambda a: ('pole' not in a and 'near_pole' not in a,
                                            -np.linalg.norm(a['end']-a['start'])))
    # Cap each family independently so many parallel pairs do not displace gap searches.
    for kind in ('gap', 'pole', 'parallel'):
        for axis in [a for a in ranked_axes if a['kind']==kind][:180]:
            candidate = follow_ridge(gray, distance, axis)
            if candidate:
                proposals.append(candidate)
    selected = []
    rank = lambda c: -(c['score'] * (1.12 if 'pole' in c or 'near_pole' in c else 1))
    def collect(items, maximum):
        for c in sorted(items, key=rank):
            if len(selected) >= maximum:
                break
            if not any(similarity(c['xy'], p['xy']) >= .65 for p in selected):
                selected.append(c)
    for kind in ('gap', 'pole', 'parallel'):
        collect([c for c in proposals if c['kind']==kind], min(limit, len(selected)+limit//3))
    collect(proposals, limit)
    return [dict(c, id=f'S{i:02d}') for i,c in enumerate(selected, 1)], dict(
        seeds=len(seeds), corridors=len(axes), ridge_candidates=len(proposals))
