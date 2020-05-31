from PIL import ImageDraw


def draw_image(image, boxes, labels=None, scores=None):
    """

    :param image: PIL.Image.Image
    :param boxes: numpy ndarray (N, 4): (x1, y1, x2, y2)
    :param labels: numpy ndarray (N,)
    :param scores: numpy ndarray (N,)
    :return:
    """
    draw = ImageDraw.Draw(image, mode="RGBA")
    N = len(boxes)
    for i in range(N):
        x1, y1, x2, y2 = boxes[i]
        # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        xy = [(x1, y1), (x2, y2)]
        draw.rectangle(xy, fill=(0, 0, 0, 0), outline=(0xff, 0, 0, 0xff))

    return image
