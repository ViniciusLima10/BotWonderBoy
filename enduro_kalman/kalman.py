import numpy as np
import cv2

class KalmanTrackedObject:
    def __init__(self, x, y):
        self.kalman = cv2.KalmanFilter(4, 2)  # 4D state, 2D measurement

        # State: [x, y, vx, vy]
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

        self.kalman.statePre = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.kalman.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)

    def update(self, x, y):
        """Update the filter with new observed position."""
        self.kalman.correct(np.array([[x], [y]], dtype=np.float32))

    def predict(self):
        """Predict the next position."""
        predicted = self.kalman.predict()
        return (predicted[0][0], predicted[1][0])  # x, y

    def current(self):
        return (self.kalman.statePost[0][0], self.kalman.statePost[1][0])
