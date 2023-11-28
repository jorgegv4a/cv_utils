import numpy as np
import uuid
import cv2

from enum import Enum, auto
from typing import Optional, List, Any

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
    clipped_abs_x0, clipped_abs_y0, clipped_abs_x0, clipped_abs_y1 are the viewing extends, the coordinates of the full image we are sampling
    height, width are the dimensions of the full image
    zoom_scale equals the magnification of the full image (ie, >1 is zooming in)

    """
    def __init__(self, full_image: np.ndarray, canvas_height: int, canvas_width: int, window_title: Optional[str] = None):
        self.full_image: np.ndarray = full_image

        self.mini_image_scale = max(self.full_image.shape) / 800
        self.mini_image: np.ndarray = cv2.resize(full_image, None, fx=1 / self.mini_image_scale, fy=1 / self.mini_image_scale)

        self.max_canvas_height: int = canvas_height
        self.max_canvas_width: int = canvas_width

        self.ar_scale = max(self.full_image.shape) / max(self.max_canvas_width, self.max_canvas_height)

        self.zoom_scale: float = 1
        self.zoom_delta: float = 1.25
        self.zoom_min: float = 1 / self.ar_scale
        self.zoom_max: float = 100 * self.ar_scale

        self.zoom_dynamic_range: float = self.zoom_max / self.zoom_min
        self.mipmaps: List[np.ndarray] = [self.full_image.copy()]
        for i in range(1, np.floor(np.log2(self.zoom_dynamic_range) - 1).astype(int)):
            self.mipmaps.append(cv2.pyrDown(self.mipmaps[-1]).copy())

        self.delta = 0.7
        self.image_x: float = 0.5
        self.image_y: float = 0.5

        self.mouse_x: Optional[int] = None
        self.mouse_y: Optional[int] = None

        self.interp_order = 0
        self.background_color = 255 if len(full_image.shape) < 3 else (0, 80, 0)

        self.last_frame_pos = None

        self.window: str = None
        self.mini_window: str = None
        self.window_title: str = None
        self.mini_window_title: str = None
        self.init_window(window_title)

    @property
    def image_abs_x(self) -> int:
        return int(self.image_x * self.width)

    @property
    def image_abs_y(self) -> int:
        return int(self.image_y * self.height)

    @property
    def image_abs_x0(self) -> int:
        return int(self.image_x0 * self.width)

    @property
    def image_abs_x1(self) -> int:
        return int(self.image_x1 * self.width)

    @property
    def image_abs_y0(self) -> int:
        return int(self.image_y0 * self.height)

    @property
    def image_abs_y1(self) -> int:
        return int(self.image_y1 * self.height)

    @property
    def image_x0(self) -> float:
        return self.image_x - ((self.max_canvas_width * self.ar_scale / self.zoom_scale) / self.width) / 2

    @property
    def image_y0(self) -> float:
        return self.image_y - ((self.max_canvas_height * self.ar_scale / self.zoom_scale) / self.height) / 2

    @property
    def image_x1(self) -> float:
        return self.image_x + ((self.max_canvas_width * self.ar_scale / self.zoom_scale) / self.width) / 2

    @property
    def image_y1(self) -> float:
        return self.image_y + ((self.max_canvas_height * self.ar_scale / self.zoom_scale) / self.height) / 2

    @property
    def scale(self) -> float:
        return self.ar_scale / self.zoom_scale

    @property
    def clipped_x0(self) -> float:
        return clip(self.image_x0, 0, 1)

    @property
    def clipped_y0(self) -> float:
        return clip(self.image_y0, 0, 1)

    @property
    def clipped_x1(self) -> float:
        return clip(self.image_x1, 0, 1)

    @property
    def clipped_y1(self) -> float:
        return clip(self.image_x1, 0, 1)

    @property
    def clipped_abs_x0(self) -> int:
        return clip(self.image_abs_x0, 0, self.width - 1)

    @property
    def clipped_abs_y0(self) -> int:
        return clip(self.image_abs_y0, 0, self.height - 1)

    @property
    def clipped_abs_x1(self) -> int:
        return clip(self.image_abs_x1, 0, self.width - 1)

    @property
    def clipped_abs_y1(self) -> int:
        return clip(self.image_abs_y1, 0, self.height - 1)

    @property
    def sampled_width(self) -> int:
        return self.image_abs_x1 - self.image_abs_x0 + 1

    @property
    def sampled_height(self) -> int:
        return self.image_abs_y1 - self.image_abs_y0 + 1

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
    def mini_height(self) -> int:
        return self.mini_image.shape[0]

    @property
    def mini_width(self) -> int:
        return self.mini_image.shape[1]

    @property
    def ar(self) -> float:
        return self.width / self.height

    @property
    def mip_level(self) -> int:
        # essentially, get how many times we need to divide the image by 2 to get to the max canvas size
        return max(0, np.min(np.floor(np.log2(np.array([self.sampled_width, self.sampled_height]) / np.array([self.max_canvas_width, self.max_canvas_height])))).astype(int))

    def recenter_canvas_view(self, x: float, y: float):
        self.image_x = x
        self.image_y = y

    def mouse_cb(self, event: int, canvas_abs_x: int, canvas_abs_y: int, flags: int, param: Any):
        if event == cv2.EVENT_LBUTTONDOWN:
            new_x = self.canvas_abs_x_to_image_x(canvas_abs_x)
            new_y = self.canvas_abs_y_to_image_y(canvas_abs_y)
            self.last_frame_pos = (new_x, new_y)
        #     new_x = self.canvas_abs_x_to_image_x(canvas_abs_x)
        #     new_y = self.canvas_abs_y_to_image_y(canvas_abs_y)
        #     self.mouse_x = new_x
        #     self.mouse_y = new_y
        #     self.recenter_canvas_view(new_x, new_y)
        elif event == cv2.EVENT_MOUSEMOVE:
            new_x = self.canvas_abs_x_to_image_x(canvas_abs_x)
            new_y = self.canvas_abs_y_to_image_y(canvas_abs_y)
            if flags & cv2.EVENT_FLAG_LBUTTON:
                if self.last_frame_pos is not None:
                    delta_x = new_x - self.last_frame_pos[0]
                    delta_y = new_y - self.last_frame_pos[1]
                    self.image_x -= delta_x
                    self.image_y -= delta_y
                else:
                    self.last_frame_pos = (new_x, new_y)

            self.mouse_x = new_x
            self.mouse_y = new_y

    def mini_window_mouse_cb(self, event: int, mini_abs_x: int, mini_abs_y: int, flags: int, param: Any):
        if event == cv2.EVENT_LBUTTONDOWN:
            new_x = self.mini_window_abs_x_to_image_x(mini_abs_x)
            new_y = self.mini_window_abs_y_to_image_y(mini_abs_y)
            self.recenter_canvas_view(new_x, new_y)
        elif event == cv2.EVENT_MOUSEMOVE:
            mini_abs_x = clip(mini_abs_x, 0, self.mini_width - 1)
            mini_abs_y = clip(mini_abs_y, 0, self.mini_height - 1)
            if flags & cv2.EVENT_FLAG_LBUTTON:
                new_x = self.mini_window_abs_x_to_image_x(mini_abs_x)
                new_y = self.mini_window_abs_y_to_image_y(mini_abs_y)
                self.recenter_canvas_view(new_x, new_y)

    def manage_input(self, input_type: Input):
        if input_type == Input.MOVE_UP:
            self.image_y = clip(self.image_y - self.delta * self.max_canvas_height * self.scale / self.height, 0, 1)
        elif input_type == Input.MOVE_LEFT:
            self.image_x = clip(self.image_x - self.delta * self.max_canvas_width * self.scale / self.width, 0, 1)
        elif input_type == Input.MOVE_DOWN:
            self.image_y = clip(self.image_y + self.delta * self.max_canvas_height * self.scale / self.height, 0, 1)
        elif input_type == Input.MOVE_RIGHT:
            self.image_x = clip(self.image_x + self.delta * self.max_canvas_width * self.scale / self.width, 0, 1)

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

        self.mini_window = f"[Mini] Image Viewer {uuid.uuid4()}"
        if window_title is None:
            self.mini_window_title = "[Mini] Image Viewer"
        cv2.namedWindow(self.window)
        cv2.setWindowTitle(self.window, self.window_title)
        cv2.setMouseCallback(self.window, self.mouse_cb)

        cv2.namedWindow(self.mini_window)
        cv2.setWindowTitle(self.mini_window, self.mini_window_title)
        cv2.setMouseCallback(self.mini_window, self.mini_window_mouse_cb)

    def canvas_abs_x_to_image_abs_x(self, canvas_abs_x: int) -> int:
        return self.image_abs_x0 + int(canvas_abs_x * self.scale)

    def canvas_abs_x_to_image_x(self, canvas_abs_x: int) -> float:
        return self.canvas_abs_x_to_image_abs_x(canvas_abs_x) / self.width

    def canvas_abs_y_to_image_abs_y(self, canvas_abs_y: int) -> int:
        return self.image_abs_y0 + int(canvas_abs_y * self.scale)

    def canvas_abs_y_to_image_y(self, canvas_abs_y: int) -> float:
        return self.canvas_abs_y_to_image_abs_y(canvas_abs_y) / self.height

    def image_abs_x_to_canvas_abs(self, image_abs_x: int) -> int:
        return int((image_abs_x - self.image_abs_x0) / self.scale)

    def image_abs_y_to_canvas_abs(self, image_abs_y: int) -> int:
        return int((image_abs_y - self.image_abs_y0) / self.scale)

    def mini_window_abs_x_to_image_abs_x(self, mini_window_abs_x: int) -> int:
        return int(self.mini_window_abs_x_to_image_x(mini_window_abs_x) * self.width)

    def mini_window_abs_x_to_image_x(self, mini_window_abs_x: int) -> float:
        return mini_window_abs_x / self.mini_width

    def mini_window_abs_y_to_image_abs_y(self, mini_window_abs_y: int) -> int:
        return int(self.mini_window_abs_y_to_image_y(mini_window_abs_y) * self.height)

    def mini_window_abs_y_to_image_y(self, mini_window_abs_y: int) -> float:
        return mini_window_abs_y / self.mini_height

    def get_viewing_crop(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        abs_x0 = int(self.image_x0 * width)
        abs_x1 = int(self.image_x1 * width)
        abs_y0 = int(self.image_y0 * height)
        abs_y1 = int(self.image_y1 * height)

        clip_x0 = clip(abs_x0, 0, width)
        clip_x1 = clip(abs_x1, 0, width)
        clip_y0 = clip(abs_y0, 0, height)
        clip_y1 = clip(abs_y1, 0, height)

        crop = image[clip_y0: clip_y1, clip_x0: clip_x1]

        left_pad = max(0, -abs_x0)
        top_pad = max(0, -abs_y0)
        right_pad = max(0, abs_x1 - width + 1)
        bottom_pad = max(0, abs_y1 - height + 1)

        padded_image = cv2.copyMakeBorder(crop, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        return padded_image

    def display(self):
        view_width = self.max_canvas_width
        view_height = self.max_canvas_height

        crop = self.get_viewing_crop(self.mipmaps[self.mip_level])

        view_image = interp_2d_to_new_shape(crop, (view_height, view_width), order=self.interp_order)
        showable_image = view_image

        mini_image = self.mini_image.copy()
        rectangle(mini_image, (self.clipped_abs_x0 // self.mini_image_scale, self.clipped_abs_y0 // self.mini_image_scale), (self.clipped_abs_x1 // self.mini_image_scale, self.clipped_abs_y1 // self.mini_image_scale))
        put_text(mini_image, f"Zoom: {self.zoom_scale:.2f} | Mip: {self.mip_level}", font_scale=0.5, color=(0, 0, 255))
        if self.mouse_x and self.mouse_y:
            put_text(mini_image, f"({self.mouse_x:.5f}, {self.mouse_y:.5f})", (0, 40), font_scale=0.5, color=(0, 0, 255))
        cv2.imshow(self.mini_window, mini_image)

        cv2.imshow(self.window, showable_image)
        key = cv2.waitKey(1)
        self.key_cb(key)


def main():
    sample_img = cv2.imread("checkerboard_v.png")
    viewer = LargeImageViewer(sample_img, 800, 1024)
    while True:
        viewer.display()


if __name__ == "__main__":
    main()