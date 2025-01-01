# src/main.py
import sys
import logging
import numpy as np
import torch
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer, QObject, pyqtSignal

from variable_viewer import \
    VariableViewer  # Ensure this module is correctly implemented
from variables import Variables  # Ensure this module is correctly implemented

# Configure logging for debugging purposes
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to see detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    # Instantiate the Variables class
    app_variables = Variables()

    # Enhanced Test data with custom objects
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

    # Determine the device for torch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Add variables to the Variables instance
    app_variables.set_variable("list_var", [1, 2, 3, 4, 5])
    app_variables.set_variable("nested_dict", {"a": 1, "b": {"c": 2, "d": 3}})
    app_variables.set_variable("numpy_array", np.random.rand(100, 100))
    app_variables.set_variable("torch_tensor", torch.rand(10, 10))
    app_variables.set_variable("string_var", "Try right click update!")
    app_variables.set_variable("test_obj", person_john)  # Custom object
    app_variables.set_variable("list_obj", [person_john, person_john, person_john])
    app_variables.set_variable("huge_tensor", torch.rand(10000, 10000).to(device))
    app_variables.set_variable("complex_nested_dict", {
        "level1": {
            "level2": {
                "level3": [1, 2, 3, 4, {"deep": "value"}]
            }
        }
    })
    app_variables.set_variable("cyclic_ref", {})
    app_variables.cyclic_ref[
        "self"] = app_variables.cyclic_ref  # Establish cyclic reference

    # Initialize the QApplication
    app = QApplication(sys.argv)

    # Instantiate the VariableViewer with the Variables instance
    viewer = VariableViewer(app_variables)
    viewer.add_console(app_variables)  # Integrate the console
    viewer.show()

    def update_variables_after_show(variables_instance):
        """Update variables once the GUI is displayed."""
        logging.info("Scheduled update triggered.")
        variables_instance.string_var = "Updated String"
        variables_instance.nested_dict["a"] = "updated"
        # Modify custom object
        variables_instance.test_obj.age = 31
        variables_instance.test_obj.car.engine.horsepower = 500
        logging.info("Variables have been updated after GUI display.")

    # Schedule the modifications to happen once the event loop starts (after 5 seconds)
    QTimer.singleShot(5000, lambda: update_variables_after_show(app_variables))

    # Execute the application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
