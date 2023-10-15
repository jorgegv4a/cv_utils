import cv2
import skimage
import numpy as np

from typing import Tuple, Optional, List, Callable, TYPE_CHECKING
from PIL import Image

from general import args_to_kwargs, default
from math_utils import clip


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


def line(img, pt1, pt2=None, color=None, thickness=None, line_type=None, shift=None):
    """
    A wrapper for cv2.line: converts points to int. If only one of the points is given, asumes vertical/horizontal line.
    Takes the same arguments as cv2.line, will add default values to any argument beyond :pt2:
    :param img:
    :param pt1:
    :param pt2:
    :param color:
    :param thickness:
    :param line_type:
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
    line_type = default(line_type, cv2.LINE_AA)
    shift = default(shift, 0)

    cv2.line(img, pt1, pt2, color, thickness, line_type, shift)


def circle(img, center, radius=None, color=None, thickness=None, line_type=None, shift=None):
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
    line_type = default(line_type, cv2.LINE_AA)
    shift = default(shift, 0)

    cv2.circle(img, center, radius, color, thickness, line_type, shift)


def rectangle(img, pt1, pt2, color=None, thickness=None, line_type=None, shift=None):
    """
    A wrapper for cv2.rectangle: converts points to int.
    Takes the same arguments as cv2.rectangle, will add default values to any argument beyond :pt2:
    :param img:
    :param pt1:
    :param pt2:
    :param color:
    :param thickness:
    :param line_type:
    :param shift:
    :return:
    """
    pt1 = tuple([int(x) for x in pt1])
    pt2 = tuple([int(x) for x in pt2])
    color = default(color, (0, 0, 255))
    thickness = default(thickness, 2)
    line_type = default(line_type, cv2.LINE_AA)
    shift = default(shift, 0)

    cv2.rectangle(img, pt1, pt2, color, thickness, line_type, shift)


def put_text(img, text, org=None, font_face=None, font_scale=None, color=None, thickness=None, line_type=None):
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
    font_face = default(font_face, cv2.FONT_HERSHEY_SIMPLEX)
    font_scale = default(font_scale, 1)
    color = default(color, (0, 0, 255))
    thickness = default(thickness, 2)
    line_type = default(line_type, cv2.LINE_AA)
    text = str(text)

    (text_w, text_h), baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
    if org is None or org[1] < text_h:
        org = (0, text_h)
        text_h *= 2
    else:
        org = [int(x) for x in org]

    cv2.putText(img, text, org, font_face, font_scale, color, thickness, line_type)

    return text_w, text_h


def imshow(*args, destroy: bool=True, delay: int=0):
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

    cv2.imshow(window_name, img)
    value = cv2.waitKey(delay)
    if destroy:
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


def window_visible(window_name: str):
    """
    Checks whether a cv2 window exists
    :param window_name:
    :return:
    """
    try:
        visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1
    except:
        visible = False
    return visible


class Trackbar:
    def __init__(self, track_name: str, value: float, value_min: float, value_max: float, num_steps: int=100, custom_cb: Optional[Callable]=None):
        """
        Helper class for creating variables whose value changes by a cv2 trackbar.
        "Native scale" being the arbitrary units for the variable are mapped to "int scale" on the slider.

        :param track_name: trackbar name
        :param value: initial value, in native scale. Will be adjusted to the closes possible int on the slider
        for the given resolution
        :param value_min: minimum value in native units
        :param value_max: maximum value in native units
        :param num_steps: Number of positions on the slider, determines the resolution. Defaults to 100
        :param custom_cb: Optional callback expecting a single arg for the new value of the variable in native units
        """
        self.track_name: str = track_name
        self._start_value: float = value
        self.value: float = value
        self.value_min: float = value_min
        self.value_max: float = value_max
        self.num_steps: int = num_steps
        self.custom_cb = custom_cb

        self.step_size = (self.value_max - self.value_min) / self.num_steps

    def as_int(self, value: float):
        return clip(round((value - self.value_min) / self.step_size), 0, self.num_steps)

    def as_float(self, value: int):
        return value * self.step_size + self.value_min

    @property
    def start_value(self):
        return self.as_int(self._start_value)

    def callback(self, new_value_int: int):
        value = self.as_float(new_value_int)
        if self.custom_cb:
            value = self.custom_cb(value)
        self.value = value


def add_trackbar(window_name: str, trackbar: Trackbar):
    """
    Adds a Trackbar for a named window. Creates the window if it doesn't exist
    :param window_name: window name
    :param trackbar: Trackbar object to be added
    :return:
    """
    if not window_visible(window_name):
        cv2.namedWindow(window_name)

    cv2.createTrackbar(trackbar.track_name, window_name, trackbar.start_value, trackbar.num_steps, trackbar.callback)


def interp_2d_to_new_shape(old_array: np.ndarray, new_shape: Tuple[int, int], order: int=3) -> np.ndarray:
    """
    Interpolate a 2d array / img to a new shape
    :param old_array:
    :param new_shape: (height, width)
    :return:
    """
    return skimage.transform.resize(old_array, new_shape, order=order)


def array2d_to_image(data: np.ndarray, log_scale: bool=False) -> np.ndarray:
    """
    Applies min-max normalization on :data: and remaps to np.uint8 range to create an image cv2 can show
    :param data: image data
    :param log_scale: if True will apply log compression
    :return:
    """
    data = (data - data.min()) / (data.max() - data.min())
    if log_scale:
        data = np.log(data + 1) / np.log(2)
    data = (data * 255).astype(np.uint8)
    return data


def pil_to_cv2(image: 'Image'):
    """
    Convert a PIL Image to cv2 format (BGR np.ndarray)
    :param image:
    :return:
    """
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def np_to_pil(image):
    """
    Convert a cv2 array (BGR np.dnarray) to PIL.Image
    :param image:
    :return:
    """
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    return im_pil


def rgb_to_hex_string(color: Tuple[int, int, int]):
    """
    Get hex string for a given RGB color tuple
    :param color:
    :return:
    """
    return '#' + ''.join([f'{ch:02x}' for ch in color])


def bgr_to_hex_string(color: Tuple[int, int, int]):
    """
    Get hex string for a given BGR color tuple
    :param color:
    :return:
    """
    return rgb_to_hex_string(color[::-1])


if __name__ == '__main__':
    # hue = hue_from_range(0.5)
    # hsv_to_bgr(hue)
    # cvcolor_from_range(7, 14, min_value=0, start_hue=0, end_hue=179, as_tuple=True)
    img = cv2.imread("/home/mojonero/andres_concentrao.jpg")
    img = resize(img, fx=0.3)
    r_tb = Trackbar("r", 1.0, 0, 1.0)
    g_tb = Trackbar("g", 1.0, 0, 1.0)
    b_tb = Trackbar("b", 1.0, 0, 1.0)
    add_trackbar("andres", r_tb)
    add_trackbar("andres", g_tb)
    add_trackbar("andres", b_tb)
    while True:
        img_cp = img.copy()
        img_cp[..., 0] = (img_cp[..., 0] * b_tb.value).astype(np.uint8)
        img_cp[..., 1] = (img_cp[..., 1] * g_tb.value).astype(np.uint8)
        img_cp[..., 2] = (img_cp[..., 2] * r_tb.value).astype(np.uint8)
        imshow("andres", img_cp, destroy=False, delay=1)
