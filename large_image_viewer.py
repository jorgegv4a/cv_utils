import numpy as np
import uuid
import cv2

from enum import Enum, auto
from threading import Thread
from typing import Tuple, Optional, List, Any

from math_utils import clip
from computer_vision import imshow, interp_2d_to_new_shape, rectangle, put_text


class Input(Enum):
    MOVE_UP = auto()
    MOVE_DOWN = auto()
    MOVE_LEFT = auto()
    MOVE_RIGHT = auto()
    ZOOM_IN = auto()
    ZOOM_OUT = auto()
    RESET = auto()
    QUIT = auto()


class LargeImageViewer:
    """
    x0, y0, x0, y1 are the viewing extends, the coordinates of the full image we are sampling
    height, width are the dimensions of the full image
    zoom_scale equals the magnification of the full image (ie, >1 is zooming in)

    """
    def __init__(self, full_image: np.ndarray, canvas_height: int, canvas_width: int, window_title: Optional[str] = None):
        self.full_image: np.ndarray = full_image
        self.side_image_scale = max(self.full_image.shape) / 800
        self.mini_image: np.ndarray = cv2.resize(full_image, None, fx=1 / self.side_image_scale, fy=1 / self.side_image_scale)

        self.max_canvas_height: int = canvas_height
        self.max_canvas_width: int = canvas_width

        repr_ratio = max(self.full_image.shape) / max(self.max_canvas_width, self.max_canvas_height)

        self.zoom_scale: float = 1
        self.zoom_delta: float = 1.25
        self.zoom_min: float = 1 / repr_ratio
        self.zoom_max: float = 10 * repr_ratio

        self.delta = 0.7
        self.x: int = self.width // 2
        self.y: int = self.height // 2

        self.mouse_x: Optional[int] = None
        self.mouse_y: Optional[int] = None

        self.interp_order = 0
        self.background_color = 255 if len(full_image.shape) < 3 else (0, 80, 0)

        self.window: str = None
        self.side_window: str = None
        self.window_title: str = None
        self.side_window_title: str = None
        self.init_window(window_title)

    @property
    def left_pad(self) -> int:
        return max(0, -self.raw_x0)

    @property
    def top_pad(self) -> int:
        return max(0, -self.raw_y0)

    @property
    def right_pad(self) -> int:
        return max(0, self.raw_x1 - self.width + 1)

    @property
    def bottom_pad(self) -> int:
        return max(0, self.raw_y1 - self.height + 1)

    @property
    def raw_x0(self) -> int:
        return self.x - int(self.max_canvas_width * self.scale)

    @property
    def raw_x1(self) -> int:
        return self.x + int(self.max_canvas_width * self.scale)

    @property
    def raw_y0(self) -> int:
        return self.y - int(self.max_canvas_height * self.scale)

    @property
    def raw_y1(self) -> int:
        return self.y + int(self.max_canvas_height * self.scale)

    @property
    def scale(self) -> float:
        if self.is_wide:
            return (self.half_width / self.max_canvas_width) / self.zoom_scale
        else:
            return (self.half_height / self.max_canvas_height) / self.zoom_scale

    @property
    def x0(self) -> int:
        return clip(self.raw_x0, 0, self.width - 1)

    @property
    def y0(self) -> int:
        return clip(self.raw_y0, 0, self.height - 1)

    @property
    def x1(self) -> int:
        return clip(self.raw_x1, 0, self.width - 1)

    @property
    def y1(self) -> int:
        return clip(self.raw_y1, 0, self.height - 1)

    @property
    def height(self) -> int:
        return self.full_image.shape[0]

    @property
    def width(self) -> int:
        return self.full_image.shape[1]

    @property
    def half_width(self) -> int:
        return self.width // 2

    @property
    def half_height(self) -> int:
        return self.height // 2

    @property
    def ar(self) -> float:
        return self.width / self.height

    @property
    def is_wide(self) -> bool:
        return self.ar > 1

    def mouse_cb(self, event: int, local_x: int, local_y: int, flags: int, param: Any):
        raw_x = self.local_x_to_raw(local_x)
        raw_y = self.local_y_to_raw(local_y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.x = raw_x
            self.y = raw_y

            self.mouse_x = raw_x
            self.mouse_y = raw_y
        elif event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x = raw_x
            self.mouse_y = raw_y

    def manage_input(self, input_type: Input):
        if input_type == Input.MOVE_UP:
            self.y = clip(self.y - int(self.delta * self.max_canvas_height * self.scale), 0, self.height)
        elif input_type == Input.MOVE_LEFT:
            self.x = clip(self.x - int(self.delta * self.max_canvas_width * self.scale), 0, self.width)
        elif input_type == Input.MOVE_DOWN:
            self.y = clip(self.y + int(self.delta * self.max_canvas_height * self.scale), 0, self.height)
        elif input_type == Input.MOVE_RIGHT:
            self.x = clip(self.x + int(self.delta * self.max_canvas_width * self.scale), 0, self.width)

        elif input_type == Input.ZOOM_IN:
            self.zoom_scale = clip(self.zoom_scale * self.zoom_delta, self.zoom_min, self.zoom_max)
        elif input_type == Input.ZOOM_OUT:
            self.zoom_scale = clip(self.zoom_scale / self.zoom_delta, self.zoom_min, self.zoom_max)

    def key_cb(self, key: int):
        if key == ord("a"):
            self.manage_input(Input.MOVE_LEFT)
        elif key == ord("d"):
            self.manage_input(Input.MOVE_RIGHT)
        elif key == ord("w"):
            self.manage_input(Input.MOVE_UP)
        elif key == ord("s"):
            self.manage_input(Input.MOVE_DOWN)
        elif key in (43, 171):
            self.manage_input(Input.ZOOM_IN)
        elif key in (45, 173):
            self.manage_input(Input.ZOOM_OUT)

    def init_window(self, window_title):
        self.window = f"Image Viewer {uuid.uuid4()}"
        if window_title is None:
            self.window_title = "Image Viewer"

        self.side_window = f"[Side] Image Viewer {uuid.uuid4()}"
        if window_title is None:
            self.side_window_title = "[Side] Image Viewer"
        cv2.namedWindow(self.window)
        cv2.setWindowTitle(self.window, self.window_title)
        cv2.setMouseCallback(self.window, self.mouse_cb)

        cv2.namedWindow(self.side_window)
        cv2.setWindowTitle(self.side_window, self.side_window_title)

    def local_x_to_raw(self, local_x: int) -> int:
        return self.raw_x0 + int(local_x * 2 * self.scale)

    def local_y_to_raw(self, local_y: int) -> int:
        return self.raw_y0 + int(local_y * 2 * self.scale)

    def raw_x_to_local(self, raw_x: int) -> int:
        return int((raw_x - self.raw_x0) / (2 * self.scale))

    def raw_y_to_local(self, raw_y: int) -> int:
        return int((raw_y - self.raw_y0) / (2 * self.scale))

    def pad_crop(self, crop) -> np.ndarray:
        padded_image = cv2.copyMakeBorder(crop, self.top_pad, self.bottom_pad, self.left_pad, self.right_pad, cv2.BORDER_CONSTANT, value=(0, 0, 255))
        return padded_image

    def display(self):
        crop = self.full_image[self.y0: self.y1, self.x0: self.x1]
        view_width = self.max_canvas_width
        view_height = self.max_canvas_height

        crop = self.pad_crop(crop)

        view_image = interp_2d_to_new_shape(crop, (view_height, view_width), order=self.interp_order)
        showable_image = view_image

        side_image = self.mini_image.copy()
        rectangle(side_image, (self.x0 // self.side_image_scale, self.y0 // self.side_image_scale), (self.x1 // self.side_image_scale, self.y1 // self.side_image_scale))
        put_text(side_image, f"Zoom: {self.zoom_scale:.2f}", font_scale=0.5, color=(0, 0, 255))
        if self.mouse_x and self.mouse_y:
            put_text(side_image, f"({self.mouse_x:>5}, {self.mouse_y:>5})", (0, 40), font_scale=0.5, color=(0, 0, 0))
        cv2.imshow(self.side_window, side_image)

        cv2.imshow(self.window, showable_image)
        key = cv2.waitKey(1)
        self.key_cb(key)


def main():
    sample_img = cv2.imread("/home/mojonero/PycharmProjects/lukeScrolls/monster/slices/40.tif")
    viewer = LargeImageViewer(sample_img, 800, 1024)
    while True:
        viewer.display()


if __name__ == "__main__":
    main()