import torch


def compute_regression_targets(locations, box, stripe_width=10):
    import pdb; pdb.set_trace()
    xs, ys = locations[:, 0], locations[:, 1]

    l = xs - box[0]
    t = ys - box[1]
    r = box[2] - xs
    b = box[3] - ys

    # build left stripe
    mask = ((torch.abs(l) <= stripe_width) * (t >= 0) * (b >= 0)).float()
    lc = (stripe_width - torch.abs(l)) * mask  # presence mask
    l = l * mask  # regression mask

    # build top stripe
    mask = ((torch.abs(t) <= stripe_width) * (l >= 0) * (r >= 0)).float()
    tc = (stripe_width - torch.abs(t)) * mask  # presence mask
    t = t * mask

    # build right stripe
    mask = ((torch.abs(r) <= stripe_width) * (t >= 0) * (b >= 0)).float()
    rc = (stripe_width - torch.abs(r)) * mask  # presence mask
    r = r * mask

    # build bottom stripe
    mask = ((torch.abs(b) <= stripe_width) * (l >= 0) * (r >= 0)).float()
    bc = (stripe_width - torch.abs(b)) * mask  # presence mask
    b = b * mask

    foc = torch.stack([lc, tc, rc, bc], dim=1)
    reg = torch.stack([l, t, r, b], dim=1)

    return {
        'focus': foc,
        'regression': reg,
    }


if __name__ == '__main__':
    box = torch.tensor([
        10, 20, 300, 400
    ]).float()

    locations = torch.tensor([
        [5, 5],
        [5, 15],
        [15, 5],
        [15, 15],
        [300, 400],
    ]).float()

    compute_regression_targets(locations, box)
