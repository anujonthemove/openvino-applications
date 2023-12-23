import threading
import time

import cv2
import numpy as np
from shapely.geometry import Point, Polygon


def put_text_with_background(
    image, instructions, cv_font=cv2.FONT_HERSHEY_PLAIN, position="top-right"
):
    if image is None:
        raise ValueError("Input image is None.")

    font = cv_font
    image_with_instructions = image.copy()
    height, width, _ = image_with_instructions.shape

    text_size = cv2.getTextSize(max(instructions, key=len), font, 1, 2)[0]
    padding = 10
    bg_height = len(instructions) * text_size[1] + padding * 2
    bg_width = text_size[0] + padding * 3

    position_mapping = {
        "top-left": (0, 0),
        "top-right": (width - bg_width, 0),
        "bottom-left": (0, height - bg_height),
        "bottom-right": (width - bg_width, height - bg_height),
        "center": ((width - bg_width) // 2, (height - bg_height) // 2),
    }

    if position not in position_mapping:
        raise ValueError(
            "Invalid position. Supported positions: 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'center'"
        )

    start_point = position_mapping[position]

    cv2.rectangle(
        image_with_instructions,
        start_point,
        (start_point[0] + bg_width, start_point[1] + bg_height),
        (0, 0, 255),
        -1,
    )

    for idx, instruction in enumerate(instructions):
        y = start_point[1] + (idx + 1) * text_size[1] + padding
        cv2.putText(
            image_with_instructions,
            instruction,
            (start_point[0] + padding, y),
            font,
            1,
            (255, 255, 255),
            1,
        )

    return image_with_instructions


class VideoPlayer:
    def __init__(self, source, fps=None):
        self.cv2 = cv2  # This is done to access the package in class methods
        self.__cap = cv2.VideoCapture(source)
        if not self.__cap.isOpened():
            raise RuntimeError(
                f"Cannot open {'camera' if isinstance(source, int) else ''} {source}"
            )
        # fps of input file
        self.__input_fps = self.__cap.get(cv2.CAP_PROP_FPS)
        if self.__input_fps <= 0:
            self.__input_fps = 60
        # target fps given by user
        self.__output_fps = fps if fps is not None else self.__input_fps
        self.__size = None

        # first frame
        _, self.__frame = self.__cap.read()
        self.__lock = threading.Lock()
        self.__thread = None
        self.__stop = False

    def start(self):
        self.__stop = False
        self.__thread = threading.Thread(target=self.__run, daemon=True)
        self.__thread.start()

    def stop(self):
        self.__stop = True
        if self.__thread is not None:
            self.__thread.join()
        self.__cap.release()

    def __run(self):
        prev_time = 0
        while not self.__stop:
            t1 = time.time()
            ret, frame = self.__cap.read()
            if not ret:
                break

            # fulfill target fps
            if 1 / self.__output_fps < time.time() - prev_time:
                prev_time = time.time()
                # replace by current frame
                with self.__lock:
                    self.__frame = frame

            t2 = time.time()
            # time to wait [s] to fulfill input fps
            wait_time = 1 / self.__input_fps - (t2 - t1)
            # wait until
            time.sleep(max(0, wait_time))

        self.__frame = None

    def next(self):
        with self.__lock:
            if self.__frame is None:
                return None
            # need to copy frame, because can be cached and reused if fps is low
            frame = self.__frame.copy()
        return frame


class PolygonDrawer:
    def __init__(self, window_name, image):
        self.FINAL_LINE_COLOR = (0, 255, 0)
        self.WORKING_LINE_COLOR = (0, 0, 255)
        self.image = image
        self.window_name = window_name
        self.done = False
        self.current = (0, 0)
        self.points = []
        self.instructions = [
            "Use mouse to select an ROI",
            " ",
            "Press 'Esc' to finish adding points",
            " ",
            "After that press any key to confirm",
        ]

    def on_mouse(self, event, x, y, buttons, user_param):
        """
        Mouse callback function that handles mouse events for polygon drawing.
        """
        if self.done:
            return
        if event == cv2.EVENT_MOUSEMOVE:
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.add_point(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.complete_polygon()

    def add_point(self, x, y):
        """
        Adds a point to the list of polygon points.
        """
        print(f"Adding point #{len(self.points)} with position ({x}, {y})")
        self.points.append((x, y))

    def complete_polygon(self):
        """
        Marks polygon drawing as complete.
        """
        print(f"Completing polygon with {len(self.points)} points.")
        self.done = True

    def draw_polygon(self, canvas):
        """
        Draws the current state of the polygon on the canvas.
        """
        if len(self.points) > 0:
            cv2.polylines(
                canvas,
                np.array([self.points]),
                False,
                self.FINAL_LINE_COLOR,
                thickness=4,
            )
            if self.current != (0, 0):
                cv2.line(
                    canvas,
                    self.points[-1],
                    self.current,
                    self.WORKING_LINE_COLOR,
                    thickness=4,
                )
        return canvas

    def run(self):
        """
        Runs the polygon drawing application.
        """
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_KEEPRATIO)
        cv2.imshow(self.window_name, self.image)
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        while not self.done:
            canvas = self.image.copy()
            # cv2.putText(canvas, "Press 'Esc' to finish adding points", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
            # cv2.putText(canvas, "Press any key to confirm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
            canvas = put_text_with_background(
                canvas, self.instructions, position="top-right"
            )
            canvas = self.draw_polygon(canvas)
            cv2.imshow(self.window_name, canvas)
            if cv2.waitKey(50) == 27:  # ESC hit
                self.done = True

        mask = np.zeros(self.image.shape[:2], np.uint8)
        if len(self.points) > 0:
            cv2.fillPoly(mask, np.array([self.points]), 255)
            self.image = cv2.bitwise_and(self.image, self.image, mask=mask)

        cv2.imshow(self.window_name, self.image)
        cv2.waitKey()
        cv2.destroyAllWindows()
        return self.image


def capture_frame_for_ROI(source):
    cap = cv2.VideoCapture(source)

    frame = None

    instructions = [
        "Press any key to get more frames",
        " ",
        "Press 's' to select frame",
        " ",
        "Press 'Esc' exit & use the visible frame",
    ]

    while True:
        ret, frame = cap.read()
        frame_copy = frame.copy()
        if not ret:
            print("Frame not found")
            break

        frame_copy = put_text_with_background(
            frame_copy, instructions, position="center"
        )
        cv2.namedWindow("frame", cv2.WINDOW_KEEPRATIO)
        cv2.imshow("frame", frame_copy)
        k = cv2.waitKey(0)
        if k == ord("s"):
            cv2.destroyAllWindows()
            return frame
        if k == 27:
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()
    cap.release()
    return frame


def create_ROI(image, write_roi=False):
    print("====> Use left click to draw polygon, right click to release and finish")
    print()
    pd = PolygonDrawer(window_name="Draw Polygonal ROI", image=image)
    result_image = pd.run()
    if write_roi:
        cv2.imwrite("polygon-roi.jpg", result_image)
    print("Polygon Points:", pd.points)
    if not pd.points:
        return None
    return pd.points


def compute_polygon_intersection(image, polygon1_points, polygon2_points):
    """
    Compute the intersection mask between two polygons and visualize it.

    Args:
        image (numpy.ndarray): The input image.
        polygon1_points (list): List of points for the first polygon.
        polygon2_points (list): List of points for the second polygon.

    Returns:
        tuple: A tuple containing:
            - polygons_intersect (bool): True if polygons intersect, False otherwise.
            - intersection_visualization (numpy.ndarray): Visualization of the intersection.
    """
    # Create a mask for both polygons
    mask1 = np.zeros_like(image, dtype=np.uint8)
    mask2 = np.zeros_like(image, dtype=np.uint8)

    cv2.fillPoly(mask1, [np.array(polygon1_points, dtype=np.int32)], (255, 255, 0))
    cv2.fillPoly(mask2, [np.array(polygon2_points, dtype=np.int32)], (255, 0, 255))

    # Compute the intersection of the two masks
    intersection_mask = cv2.bitwise_and(mask1, mask1)

    # Visualize the intersection by adding the original masks
    intersection_visualization = cv2.add(cv2.add(mask1, mask2), intersection_mask)

    # Check if the intersection mask has any non-zero pixels
    polygons_intersect = np.any(intersection_mask)

    return polygons_intersect, intersection_visualization


def is_point_inside_polygon(polygon, point) -> bool:
    """
    Check if a point is inside a polygon.

    Args:
        polygon (list[list[float]]): List of points for the polygon [[x1, y1], [x2, y2], ...].
        point (list[float]): Point [x, y].

    Returns:
        bool: True if the point is inside the polygon, False otherwise.
    """
    polygon_shape = Polygon(polygon)
    return polygon_shape.contains(Point(point))


def are_polygons_intersecting(pt1, pt2, intersection_threshold=0.5) -> bool:
    """
    Check if two polygons are intersecting.

    Args:
        pt1 (list[list[float]]): List of points for polygon 1 [[x1, y1], [x2, y2], ...].
        pt2 (list[list[float]]): List of points for polygon 2 [[x1, y1], [x2, y2], ...].
        intersection_threshold (float, optional): Intersection threshold for polygon intersection with respect to pt2. Defaults to 0.5.

    Returns:
        bool: True if the polygons are intersecting with the specified threshold, False otherwise.
    """
    polygon1 = Polygon(pt1)
    polygon2 = Polygon(pt2)

    intersection_area = polygon1.intersection(polygon2).area
    if intersection_area / polygon2.area > intersection_threshold:
        return True
    return False
