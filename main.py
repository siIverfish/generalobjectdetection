from dataclasses import dataclass
import colorsys
import random

import numpy as np
import cv2 as cv

from icecream import ic

BATCH_SIZE = 10

image_data = []


class ThresholdOptimizer:
    HUE_RANGE = 10

    def __init__(self):
        self.threshold = None

    def make_random_change(self):
        """Makes a random change to the threshold with a gamma function"""
        return Threshold(
            lower=self.threshold.lower + np.random.gamma(1, 1, 3),
            upper=self.threshold.upper + np.random.gamma(1, 1, 3),
        )

    def generate_batch(self, batch_size):
        """Generates a batch of random changes to the threshold."""
        yield from (self.make_random_change() for _ in range(batch_size))

    def get_average_rating(self, threshold):
        """Returns the average rating of the images in the image_data list."""
        return np.mean(
            [image_datapoint.get_rating(threshold) for image_datapoint in image_data]
        )
        
    def get_partial_average_rating(self, threshold, k):
        return np.mean(
            [image_datapoint.get_rating(threshold) for image_datapoint in random.choices(image_data, k=2**k)]
        )

    def evolve(self):
        """Evolves the threshold to a better one."""
        best_threshold = self.threshold
        # TODO
        best_rating = self.get_average_rating(best_threshold)
        for threshold in self.generate_batch(BATCH_SIZE):
            k = 0
            rating = self.get_partial_average_rating(threshold, k)
            while rating < best_rating:
                if 2**k > len(image_data):
                    break
                k += 1
                rating = self.get_partial_average_rating(threshold, k)
            else:
                continue
            print(
                f" ------------------ New best rating: {rating} ------------------ "
            )
            best_rating = rating
            best_threshold = threshold
        self.threshold = best_threshold

    def set_threshold_from_pixel_range(self, pixel):
        """Creates a threshold from a pixel."""
        threshold = Threshold(
            lower=np.array([pixel[0] - self.HUE_RANGE, 50, 50]),
            upper=np.array([pixel[0] + self.HUE_RANGE, 255, 255]),
        )
        self.threshold = threshold

    def set_threshold_from_bgr_pixel(self, pixel):
        """Sets the threshold."""
        # set to RGB
        pixel = pixel.tolist()
        pixel.reverse()
        # set to HSV
        pixel = np.array(pixel) / 255
        pixel = np.array(colorsys.rgb_to_hsv(*pixel))
        pixel[0] *= 180
        pixel[1] *= 255
        pixel[2] *= 255
        self.set_threshold_from_pixel_range(pixel)


@dataclass
class Threshold:
    lower: np.ndarray
    upper: np.ndarray

    def to_json(self):
        """Converts the threshold to a JSON serializable object."""
        return {"lower": self.lower.tolist(), "upper": self.upper.tolist()}

    @classmethod
    def from_json(cls, data):
        """Converts a JSON object to a threshold."""
        return cls(lower=np.array(data["lower"]), upper=np.array(data["upper"]))


@dataclass
class ImageDatapoint:
    image: np.ndarray
    center: np.ndarray

    def get_rating(self, threshold):
        central_object = get_object(self.image, threshold)
        object_center = get_contour_center(central_object)
        return point_distance(self.center, object_center)


def point_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """Returns the distance between two points."""
    return np.linalg.norm(point1 - point2)


def get_contour_center(contour):
    """Returns the center of the contour."""
    moments = cv.moments(contour)
    try:
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
    except ZeroDivisionError:
        center_x = 0
        center_y = 0
    return np.array([center_x, center_y])


def infinite_key_frame_stream():
    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        key = cv.waitKey(1)
        if ret is False or frame is None:
            continue
        if key & 0xFF == ord("q"):
            break
        yield key, frame


def get_object(image, threshold):
    """Finds the largest object in the image that is within the threshold."""
    # Blurs the image to reduce noise
    processed = cv.medianBlur(image, 5)
    # Converts the image to the Hue, Saturation, Value color space, which makes it easier to detect objects of a certain color
    processed = cv.cvtColor(processed, cv.COLOR_BGR2HSV)
    # Converts the image to a binary image, where the object of interest (between the lower and upper thresholds) is white and the rest is black
    processed = cv.inRange(processed, threshold.lower, threshold.upper)
    contours, _ = cv.findContours(processed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    largest_object = max(contours, key=cv.contourArea)
    return largest_object


class ClickHandler:
    def __init__(self, threshold_optimizer):
        self.threshold_optimizer = threshold_optimizer
        self.frame = None

    def handle_double_click(self, event, x, y, flags, param):
        """Handles double clicks on the image."""
        if self.frame is None:
            raise Exception("Frame is None")

        if event == cv.EVENT_LBUTTONDBLCLK:
            if self.threshold_optimizer.threshold is None:
                self.threshold_optimizer.set_threshold_from_bgr_pixel(self.frame[y, x])
            print(f"Clicked at ({x}, {y})")
            image_data.append(ImageDatapoint(image=self.frame, center=np.array([x, y])))


def main():
    threshold_optimizer = ThresholdOptimizer()
    click_handler = ClickHandler(threshold_optimizer)

    for key, frame in infinite_key_frame_stream():
        click_handler.frame = frame

        if threshold_optimizer.threshold is None:
            cv.imshow("frame", frame)
            cv.setMouseCallback("frame", click_handler.handle_double_click)
            continue

        center_object = get_object(frame, threshold_optimizer.threshold)
        
        if center_object is None:
            show_no_object_found(frame)
            cv.imshow("frame", frame)
            cv.setMouseCallback("frame", click_handler.handle_double_click)
            threshold_optimizer.evolve()
            continue
        
        object_center = get_contour_center(center_object)
        cv.circle(frame, tuple(object_center), 50, (0, 0, 255), -1)
        cv.imshow("frame", frame)
        cv.setMouseCallback("frame", click_handler.handle_double_click)

        threshold_optimizer.evolve()


def show_no_object_found(frame):
    """Shows a message on the frame if no object is found."""
    cv.putText(
        frame,
        "No object found",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
        cv.LINE_AA,
    )


if __name__ == "__main__":
    main()
