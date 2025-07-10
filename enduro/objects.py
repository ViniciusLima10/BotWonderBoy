class Car:
    def __init__(self, x, y, w, h):
        self.x = x  # Top-left X (in full frame coordinates)
        self.y = y  # Top-left Y
        self.w = w
        self.h = h

    def center(self):
        """Return the (x, y) of the center of the car."""
        return (self.x + self.w // 2, self.y + self.h // 2)

    def bottom(self):
        """Bottom Y position."""
        return self.y + self.h

    def __repr__(self):
        return f"Car(x={self.x}, y={self.y}, w={self.w}, h={self.h})"

class PlayerCar(Car):
    def __init__(self, x, y, w, h):
        super().__init__(x, y, w, h)

class EnemyCar(Car):
    def __init__(self, x, y, w, h):
        super().__init__(x, y, w, h)
