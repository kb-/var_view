# src/main.py
import sys
import logging
import numpy as np
import torch
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

from variable_viewer import VariableViewer  # Ensure this module is correctly implemented

# Configure logging for debugging purposes
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to see detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class AppDataSource:
    def __init__(self):
        self.list_var = [1, 2, 3, 4, 5]
        self.nested_dict = {"a": 1, "b": {"c": 2, "d": 3}}
        self.numpy_array = np.random.rand(100, 100)
        self.torch_tensor = torch.rand(10, 10)
        self.string_var = "Try right-click update!"
        self.complex_nested_dict = {
            "level1": {
                "level2": {
                    "level3": [1, 2, 3, 4, {"deep": "value"}]
                }
            }
        }
        self.cyclic_ref = {}
        self.cyclic_ref["self"] = self.cyclic_ref  # Establish cyclic reference

        # Custom objects
        class Engine:
            def __init__(self, horsepower, type_):
                self.horsepower = horsepower
                self.type = type_

            def start(self):
                return "Engine started."

        class Car:
            def __init__(self, make, model, engine):
                self.make = make
                self.model = model
                self.engine = engine
                self.owner = None  # To be set later, creating a cyclic reference

            def drive(self):
                return f"Driving the {self.make} {self.model}."

            def set_owner(self, owner):
                self.owner = owner

        class Person:
            def __init__(self, name, age, car=None):
                self.name = name
                self.age = age
                self.car = car

            def greet(self):
                return f"Hello, my name is {self.name}."

            def buy_car(self, car):
                self.car = car
                car.set_owner(self)

        # Instantiate custom objects
        engine_v8 = Engine(450, "V8")
        car_ferrari = Car("Ferrari", "488 Spider", engine_v8)
        person_john = Person("John Doe", 30)

        # Establish cyclic reference
        person_john.buy_car(car_ferrari)

        self.test_obj = person_john
        self.list_obj = [person_john, person_john, person_john]

        # Large tensor (on GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.huge_tensor = torch.rand(10000, 10000).to(device)


def main():
    # Instantiate the data source
    data_source = AppDataSource()

    # Initialize the QApplication
    app = QApplication(sys.argv)

    # Instantiate the VariableViewer with the data source
    viewer = VariableViewer(data_source, "data_source")
    viewer.add_console()  # Integrate the console
    viewer.show()

    def update_variables_after_show():
        """Update variables once the GUI is displayed."""
        logging.info("Scheduled update triggered.")
        data_source.string_var = "Updated String"
        data_source.nested_dict["a"] = "updated"
        # Modify custom object
        data_source.test_obj.age = 31
        data_source.test_obj.car.engine.horsepower = 500
        logging.info("Variables have been updated after GUI display.")

    # Schedule the modifications to happen once the event loop starts (after 5 seconds)
    QTimer.singleShot(5000, update_variables_after_show)

    # Create a timer for periodic logging
    def log_string_var():
        """Log the current value of string_var periodically."""
        logging.info(f"Periodic check: string_var = {data_source.string_var}")

    # Set up the timer to trigger every 2 seconds (2000 milliseconds)
    periodic_timer = QTimer()
    periodic_timer.timeout.connect(log_string_var)
    periodic_timer.start(2000)

    # Execute the application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
