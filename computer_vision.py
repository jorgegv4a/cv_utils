import cv2
import numpy as np
from general import args_to_kwargs, default


def hue_from_range(value, max_value=None, min_value=None, start_hue=None, end_hue=None):
    """
    Gets hue in openCV's HSV color space for a value understood to be a % between a mininum and a maximum. For example:
    hue_from_range(0.7) returns the angle in a color wheel that corresponds to 70% of a full turn (blue), starting at red and going counterclockwise.
    Note that in openCV hue ranges from 0 to 179, which this method adapts to.
    :param value: a number.
    :param max_value: the maximum value :value: could achieve. Defaults is 1.
    :param min_value: the minimum value :value: could achieve. Defaults is 0.
    :param start_hue: the starting point in the color wheel. Default is 0.
    :param end_hue: the ending point in the color wheel. Default is 179.
    :return: an int in the range [0, 179]
    """
    max_value = default(max_value, 1)
    min_value = default(min_value, 0)
    start_hue = default(start_hue, 0)
    end_hue = default(end_hue, 179)

    norm_val = (value - min_value) / (max_value - min_value)
    hue_range = end_hue - start_hue
    hue = int(start_hue + norm_val * hue_range)
    return hue


def hsv_to_bgr(hue, sat=None, value=None):
    """
    A wrapper for HSV2BGR for non images: takes a single number as input and converts it to HSV.
    :param hue: a number in the range 0 - 179.
    :param sat: Saturation. Default is 255.
    :param value: Value. Default is 255.
    :return: a numpy 1d array of [B, G, R], dtype=np.uint8
    """

    hue = int(hue)
    sat = int(default(sat, 255))
    value = int(default(value, 255))
    return cv2.cvtColor(np.array([[[hue, sat, value]]], dtype=np.uint8), cv2.COLOR_HSV2BGR).ravel()


def cvcolor_from_range(value_, max_value=None, min_value=None, start_hue=None, end_hue=None, sat=None, value=None):
    """
    Given a value, understood to be a %, returns the BGR tuple of its corresponding hue.
    :param value_:
    :param max_value:
    :param min_value:
    :param start_hue:
    :param end_hue:
    :param sat:
    :param value:
    :return:
    """
    hue = hue_from_range(value_, max_value, min_value, start_hue, end_hue)
    bgr = hsv_to_bgr(hue, sat, value)
    return tuple([int(x) for x in bgr])


def colors_from_max_hue(num_elements, max_value=None, min_value=None, start_hue=None, end_hue=None, sat=None, value=None):
    """
    Returns a list of colors sampled from HSV space such that their hue is equally spaced.
    :param num_elements:
    :param max_value:
    :param min_value:
    :param start_hue:
    :param end_hue:
    :param sat:
    :param value:
    :return:
    """
    colors = []
    for i in range(num_elements):
        color = cvcolor_from_range(i / num_elements, max_value, min_value, start_hue, end_hue, sat, value)
        colors.append(color)
    return colors


def random_full_hue(count=None):
    """
    Gets a random color from HSV with maximum saturation and value.
    :param count:
    :return:
    """
    count = default(count, 1)
    hues = np.random.uniform(size=count)
    return [cvcolor_from_range(hue) for hue in hues]


def line(img, pt1, pt2=None, color=None, thickness=None, lineType=None, shift=None):
    """
    A wrapper for cv2.line: converts points to int. If only one of the points is given, asumes vertical/horizontal line.
    Takes the same arguments as cv2.line, will add default values to any argument beyond :pt2:
    :param img:
    :param pt1:
    :param pt2:
    :param color:
    :param thickness:
    :param lineType:
    :param shift:
    :return:
    """
    try:
        pt1 = tuple([int(x) for x in pt1])
    except TypeError as e:
        if (pt1[0] or pt1[1]) is None:
            raise e
        height, width = img.shape[:2]
        if pt1[0] is not None:
            pt1 = [pt1[0], 0]
            pt2 = [pt1[0], height-1]
        else:
            pt1 = [0, pt1[1]]
            pt2 = [width - 1, pt1[1]]

    pt1 = tuple([int(x) for x in pt1])
    pt2 = tuple([int(x) for x in pt2])
    color = default(color, (0, 0, 255))
    thickness = default(thickness, 2)
    lineType = default(lineType, cv2.LINE_AA)
    shift = default(shift, 0)

    cv2.line(img, pt1, pt2, color, thickness, lineType, shift)


def circle(img, center, radius=None, color=None, thickness=None, lineType=None, shift=None):
    """
    A wrapper for cv2.circle: converts :center: to int, assumes radius 1 if unspecified.
    Takes the same arguments as cv2.circle, will add default values to any argument beyond :center:
    :param img:
    :param text:
    :param args:
    :param kwargs:
    :return:
    """
    center = tuple([int(x) for x in center])
    radius = default(radius, 1)
    color = default(color, (0, 0, 255))
    thickness = default(thickness, 2)
    lineType = default(lineType, cv2.LINE_AA)
    shift = default(shift, 0)

    cv2.circle(img, center, radius, color, thickness, lineType, shift)


def rectangle(img, pt1, pt2, color=None, thickness=None, lineType=None, shift=None):
    """
    A wrapper for cv2.rectangle: converts points to int.
    Takes the same arguments as cv2.rectangle, will add default values to any argument beyond :pt2:
    :param img:
    :param pt1:
    :param pt2:
    :param color:
    :param thickness:
    :param lineType:
    :param shift:
    :return:
    """
    pt1 = tuple([int(x) for x in pt1])
    pt2 = tuple([int(x) for x in pt2])
    color = default(color, (0, 0, 255))
    thickness = default(thickness, 2)
    lineType = default(lineType, cv2.LINE_AA)
    shift = default(shift, 0)

    cv2.rectangle(img, pt1, pt2, color, thickness, lineType, shift)


def put_text(img, text, org=None, fontFace=None, fontScale=None, color=None, thickness=None, lineType=None):
    """
    A wrapper for cv2.putText: converts origin to int, and if unspecified set the position of the text
    in the top left corner of the screen such that the full text can be seen.
    Takes the same arguments as cv2.putText, will add default values to any argument beyond :text:
    :param img:
    :param text:
    :param args:
    :param kwargs:
    :return:
    """
    org = default(org, None)
    fontFace = default(fontFace, cv2.FONT_HERSHEY_SIMPLEX)
    fontScale = default(fontScale, 1)
    color = default(color, (0, 0, 255))
    thickness = default(thickness, 2)
    lineType = default(lineType, cv2.LINE_AA)
    text = str(text)

    (text_w, text_h), baseline = cv2.getTextSize(text, fontFace, fontScale, thickness)
    if org is None or org[1] < text_h:
        org = (0, text_h)
        text_h *= 2
    else:
        org = [int(x) for x in org]

    cv2.putText(img, text, org, fontFace, fontScale, color, thickness, lineType)

    return text_w, text_h


def imshow(*args):
    """
    A wrapper for cv2.imshow: immediately show image until any input is given, allows for omission of window name
    :param window_name: string
    :return: same output as cv2.waitKey(), key number
    """
    if len(args) == 1:
        [img] = args
        window_name = "Unnamed window"
    elif len(args) == 2:
        window_name, img = args
    else:
        raise TypeError(f"imshow takes 1/2 arguments, {len(args)} given.")

    # colors = colors_from_max_hue(30)
    colors = random_full_hue(20)
    y0 = 0
    for color in colors:
        text_w, text_h = put_text(img, "Rainbows!", color=color, org=(0, y0))
        y0 += text_h

    line(img, (700, None))
    circle(img, (500, 500), 20, thickness=-1)


    cv2.imshow(window_name, img)
    value = cv2.waitKey(0)
    cv2.destroyWindow(window_name)
    return value


def resize(img, dsize=None, fx=None, fy=None, interpolation=None):
    """
    A wrapper for cv2.resize: :dsize: is cast to int, allows for only one side to be specified, preserving aspect ratio.
    Also follows the same logic if only one of :fx: and :fy: is given.
    :param img:
    :param dsize:
    :param fx:
    :param fy:
    :param interpolation:
    :return:
    """
    if (dsize or fx or fy) is None:
        raise TypeError("No output size given for resize")

    height, width = img.shape[:2]
    if dsize is None:
        if fx is None:
            fx = fy

        elif fy is None:
            fy = fx

        dsize = (int(width * fx), int(height * fy))

    elif (fx or fy) is None:
        try:
            dsize = tuple([int(x) for x in dsize])
        except TypeError as e:
            if (dsize[0] or dsize[1]) is None:
                raise e
            if dsize[0] is not None:
                fs = dsize[0] / width
                dsize = (int(dsize[0]), int(height * fs))
            else:
                fs = dsize[1] / height
                dsize = (int(width * fs), int(dsize[1]))

    resized_img = cv2.resize(img, dsize, interpolation=interpolation)
    return resized_img


if __name__ == '__main__':
    # hue = hue_from_range(0.5)
    # hsv_to_bgr(hue)
    # cvcolor_from_range(7, 14, min_value=0, start_hue=0, end_hue=179, as_tuple=True)
    img = cv2.imread("/home/mojonero/andres_concentrao.jpg")
    imshow(img)