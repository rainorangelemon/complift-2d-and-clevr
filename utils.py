from PIL import Image


def merge_pic(image_paths, column, row, save_path):
    images = [Image.open(x) for x in image_paths]
    widths, heights = zip(*(i.size for i in images))

    total_width = column * max(widths)
    total_height = row * max(heights)

    new_im = Image.new('RGB', (total_width, total_height), (255, 255, 255))

    index = 0
    for i in range(row):
        for j in range(column):
            new_im.paste(images[index], (j * max(widths), i * max(heights)))
            index += 1

    new_im.save(save_path)


def merge_pic_in_a_row(image_paths, save_path):
    images = [Image.open(x) for x in image_paths]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    total_height = max(heights)

    new_im = Image.new('RGB', (total_width, total_height), (255, 255, 255))

    offset = 0
    for im in images:
        new_im.paste(im, (offset, 0))
        offset += im.size[0]

    new_im.save(save_path)