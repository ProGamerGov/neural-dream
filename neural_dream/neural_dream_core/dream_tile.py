import torch


# Apply blend masks to tiles
def mask_tile(tile, overlap, side='bottom'):
    h, w = tile.size(2), tile.size(3)
    top_overlap, bottom_overlap, right_overlap, left_overlap = overlap[0], overlap[1], overlap[2], overlap[3]
    if tile.is_cuda:
        if 'left' in side and 'left-special' not in side:
            lin_mask_left = torch.linspace(0,1,left_overlap, device=tile.get_device()).repeat(h,1).repeat(3,1,1).unsqueeze(0)
        if 'right' in side and 'right-special' not in side:
            lin_mask_right = torch.linspace(1,0,right_overlap, device=tile.get_device()).repeat(h,1).repeat(3,1,1).unsqueeze(0)
        if 'top' in side and 'top-special' not in side:
            lin_mask_top = torch.linspace(0,1,top_overlap, device=tile.get_device()).repeat(w,1).rot90(3).repeat(3,1,1).unsqueeze(0)
        if 'bottom' in side and 'bottom-special' not in side:
            lin_mask_bottom = torch.linspace(1,0,bottom_overlap, device=tile.get_device()).repeat(w,1).rot90(3).repeat(3,1,1).unsqueeze(0)

        if 'left-special' in side:
            lin_mask_left = torch.linspace(0,1,left_overlap, device=tile.get_device())
            zeros_mask = torch.zeros(w-(left_overlap*2), device=tile.get_device())
            ones_mask = torch.ones(left_overlap, device=tile.get_device())
            lin_mask_left = torch.cat([zeros_mask, lin_mask_left, ones_mask], 0).repeat(h,1).repeat(3,1,1).unsqueeze(0)
        if 'right-special' in side:
            lin_mask_right = torch.linspace(1,0,right_overlap, device=tile.get_device())
            ones_mask = torch.ones(w-right_overlap, device=tile.get_device())
            lin_mask_right = torch.cat([ones_mask, lin_mask_right], 0).repeat(h,1).repeat(3,1,1).unsqueeze(0)
        if 'top-special' in side:
            lin_mask_top = torch.linspace(0,1,top_overlap, device=tile.get_device())
            zeros_mask = torch.zeros(h-(top_overlap*2), device=tile.get_device())
            ones_mask = torch.ones(top_overlap, device=tile.get_device())
            lin_mask_top = torch.cat([zeros_mask, lin_mask_top, ones_mask], 0).repeat(w,1).rot90(3).repeat(3,1,1).unsqueeze(0)
        if 'bottom-special' in side:
            lin_mask_bottom = torch.linspace(1,0,bottom_overlap, device=tile.get_device())
            ones_mask = torch.ones(h-bottom_overlap, device=tile.get_device())
            lin_mask_bottom = torch.cat([ones_mask, lin_mask_bottom], 0).repeat(w,1).rot90(3).repeat(3,1,1).unsqueeze(0)
    else:
        if 'left' in side and 'left-special' not in side:
            lin_mask_left = torch.linspace(0,1,left_overlap).repeat(h,1).repeat(3,1,1).unsqueeze(0)
        if 'right' in side and 'right-special' not in side:
            lin_mask_right = torch.linspace(1,0,right_overlap).repeat(h,1).repeat(3,1,1).unsqueeze(0)
        if 'top' in side and 'top-special' not in side:
            lin_mask_top = torch.linspace(0,1,top_overlap).repeat(w,1).rot90(3).repeat(3,1,1).unsqueeze(0)
        if 'bottom' in side and 'bottom-special' not in side:
            lin_mask_bottom = torch.linspace(1,0,bottom_overlap).repeat(w,1).rot90(3).repeat(3,1,1).unsqueeze(0)

        if 'left-special' in side:
            lin_mask_left = torch.linspace(0,1,left_overlap)
            zeros_mask = torch.zeros(w-(left_overlap*2))
            ones_mask = torch.ones(left_overlap)
            lin_mask_left = torch.cat([zeros_mask, lin_mask_left, ones_mask], 0).repeat(h,1).repeat(3,1,1).unsqueeze(0)
        if 'right-special' in side:
            lin_mask_right = torch.linspace(1,0,right_overlap)
            ones_mask = torch.ones(w-right_overlap)
            lin_mask_right = torch.cat([ones_mask, lin_mask_right], 0).repeat(h,1).repeat(3,1,1).unsqueeze(0)
        if 'top-special' in side:
            lin_mask_top = torch.linspace(0,1,top_overlap)
            zeros_mask = torch.zeros(h-(top_overlap*2))
            ones_mask = torch.ones(top_overlap)
            lin_mask_top = torch.cat([zeros_mask, lin_mask_top, ones_mask], 0).repeat(w,1).rot90(3).repeat(3,1,1).unsqueeze(0)
        if 'bottom-special' in side:
            lin_mask_bottom = torch.linspace(1,0,bottom_overlap)
            ones_mask = torch.ones(h-bottom_overlap)
            lin_mask_bottom = torch.cat([ones_mask, lin_mask_bottom], 0).repeat(w,1).rot90(3).repeat(3,1,1).unsqueeze(0)

    base_mask = torch.ones_like(tile)

    if 'right' in side and 'right-special' not in side:
        base_mask[:,:,:,w-right_overlap:] = base_mask[:,:,:,w-right_overlap:] * lin_mask_right
    if 'left' in side and 'left-special' not in side:
        base_mask[:,:,:,:left_overlap] = base_mask[:,:,:,:left_overlap] * lin_mask_left
    if 'bottom' in side and 'bottom-special' not in side:
        base_mask[:,:,h-bottom_overlap:,:] = base_mask[:,:,h-bottom_overlap:,:] * lin_mask_bottom
    if 'top' in side and 'top-special' not in side:
        base_mask[:,:,:top_overlap,:] = base_mask[:,:,:top_overlap,:] * lin_mask_top

    if 'right-special' in side:
        base_mask = base_mask * lin_mask_right
    if 'left-special' in side:
        base_mask = base_mask * lin_mask_left
    if 'bottom-special' in side:
        base_mask = base_mask * lin_mask_bottom
    if 'top-special' in side:
        base_mask = base_mask * lin_mask_top
    return tile * base_mask


def get_tile_coords(d, tile_dim, overlap=0):
    overlap = int(tile_dim * (1-overlap))
    c, tile_start, coords = 1, 0, [0]
    while tile_start + tile_dim < d:
        tile_start = overlap * c
        if tile_start + tile_dim >= d:
            coords.append(d - tile_dim)
        else:
            coords.append(tile_start)
        c += 1
    return coords, overlap


def get_tiles(img, tile_coords, tile_size, info_only=False):
    tile_list = []
    for y in tile_coords[0]:
        for x in tile_coords[1]:
            tile = img[:, :, y:y+tile_size[0], x:x+tile_size[1]]
            tile_list.append(tile)
    if not info_only:
        return tile_list
    else:
        return tile_list[0].size(2), tile_list[0].size(3)


def final_overlap(tile_coords):
    r, c = len(tile_coords[0]), len(tile_coords[1])
    return (tile_coords[0][r-1] - tile_coords[0][r-2], tile_coords[1][c-1] - tile_coords[1][c-2])


def add_tiles(tiles, base_img, tile_coords, tile_size, overlap):
    f_ovlp = final_overlap(tile_coords)
    h, w = tiles[0].size(2), tiles[0].size(3)
    t=0
    column, row, = 0, 0
    for y in tile_coords[0]:
        for x in tile_coords[1]:
            mask_sides=''
            c_overlap = overlap.copy()
            if len(tile_coords[0]) > 1:
                if row == 0:
                    if row == len(tile_coords[0]) - 2:
                        mask_sides += 'bottom-special'
                        c_overlap[1] = f_ovlp[0] # Change bottom overlap
                    else:
                        mask_sides += 'bottom'
                elif row > 0 and row < len(tile_coords[0]) -2:
                    mask_sides += 'bottom,top'
                elif row == len(tile_coords[0]) - 2:
                    if f_ovlp[0] > 0:
                        mask_sides += 'bottom-special,top'
                        c_overlap[1] = f_ovlp[0] # Change bottom overlap
                    elif f_ovlp[0] <= 0:
                        mask_sides += 'bottom,top'
                elif row == len(tile_coords[0]) -1:
                    if f_ovlp[0] > 0:
                        mask_sides += 'top-special'
                        c_overlap[0] = f_ovlp[0] # Change top overlap
                    elif f_ovlp[0] <= 0:
                        mask_sides += 'top'

            if len(tile_coords[1]) > 1:
                if column == 0:
                    if column == len(tile_coords[1]) -2:
                        mask_sides += ',right-special'
                        c_overlap[2] = f_ovlp[1] # Change right overlap
                    else:
                        mask_sides += ',right'
                elif column > 0 and column < len(tile_coords[1]) -2:
                    mask_sides += ',right,left'
                elif column == len(tile_coords[1]) -2:
                    if f_ovlp[1] > 0:
                        mask_sides += ',right-special,left'
                        c_overlap[2] = f_ovlp[1] # Change right overlap
                    elif f_ovlp[1] <= 0:
                        mask_sides += ',right,left'
                elif column == len(tile_coords[1]) -1:
                    if f_ovlp[1] > 0:
                        mask_sides += ',left-special'
                        c_overlap[3] = f_ovlp[1] # Change left overlap
                    elif f_ovlp[1] <= 0:
                        mask_sides += ',left'

            if t < len(tiles):
                tile = mask_tile(tiles[t], c_overlap, side=mask_sides)
                base_img[:, :, y:y+tile_size[0], x:x+tile_size[1]] = base_img[:, :, y:y+tile_size[0], x:x+tile_size[1]] + tile
            t+=1
            column+=1
        row+=1
        column=0
    return base_img


def tile_setup(tile_size, overlap_percent, base_size):
    if type(tile_size) is not tuple and type(tile_size) is not list:
        tile_size = (tile_size, tile_size)
    if type(overlap_percent) is not tuple and type(overlap_percent) is not list:
        overlap_percent = (overlap_percent, overlap_percent)
    x_coords, x_ovlp = get_tile_coords(base_size[1], tile_size[1], overlap_percent[1])
    y_coords, y_ovlp = get_tile_coords(base_size[0], tile_size[0], overlap_percent[0])
    return (y_coords, x_coords), tile_size, [y_ovlp, y_ovlp, x_ovlp, x_ovlp]


def tile_image(img, tile_size, overlap_percent, info_only=False):
    tile_coords, tile_size, _ = tile_setup(tile_size, overlap_percent, (img.size(2), img.size(3)))
    if not info_only:
        return get_tiles(img, tile_coords, tile_size)
    else:
        tile_size = get_tiles(img, tile_coords, tile_size, info_only)
        return tile_size[0], tile_size[1], (len(tile_coords[0]), len(tile_coords[1])), (len(tile_coords[0]) * len(tile_coords[1]))


def rebuild_image(tiles, base_img, tile_size, overlap_percent):
    base_img = torch.zeros_like(base_img)
    tile_coords, tile_size, overlap = tile_setup(tile_size, overlap_percent, (base_img.size(2), base_img.size(3)))
    return add_tiles(tiles, base_img, tile_coords, tile_size, overlap)
