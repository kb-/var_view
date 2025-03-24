# var_view/main.py
import sys
import logging
import numpy as np
import torch
import cv2
from collections import namedtuple
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

# Import your standard viewer and the new paginated viewer
from var_view.variable_viewer.viewer import VariableViewer

# Configure logging for debugging purposes
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to see detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.getLogger('variable_viewer').setLevel(logging.DEBUG)

class AppDataSource:
    def __init__(self):
        # Some simple variables
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

        # Create a UMat example
        self.cv2_umat = cv2.UMat(np.random.randint(0, 255, (500, 500), dtype=np.uint8))

        # Custom objects
        class Engine:
            def __init__(self, horsepower, type_):
                self.horsepower = horsepower
                self.type = type_

            def start(self):
                return "Engine started."

            def __str__(self):
                return f"Engine(type={self.type}, horsepower={self.horsepower})"

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

            # def __str__(self):
            #     return f"Car({self.make} {self.model}, Engine={self.engine})"

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

            def __str__(self):
                return f"Person({self.name}, {self.age})"

        # Instantiate custom objects
        engine_v8 = Engine(450, "V8")
        car_ferrari = Car("Ferrari", "488 Spider", engine_v8)
        person_john = Person("John Doe", 30)
        person_john.buy_car(car_ferrari)  # Establish cyclic reference

        self.test_obj = person_john
        self.list_obj = [person_john, person_john, person_john]

        # Example: Dictionaries with object keys and named tuples
        CustomKey = namedtuple('CustomKey', ['id', 'description'])
        self.custom_key = CustomKey
        self.object_key_dict = {
            CustomKey(1, "First Key"): "Object Value 1",
            CustomKey(2, "Second Key"): {"nested": [1, 2, 3]},
            car_ferrari: "Car as Key",
        }

        # New Example 1: Long multi-level list
        # Create a 3-level nested list: 1000 outer lists, each containing 10 lists,
        # each with 1000 string items.
        self.long_multi_level_list = [
            [ [f"List Item {i}-{j}-{k}" for k in range(1000)] for j in range(10)]
            for i in range(1000)
        ]

        # For demonstration, add a person_john child to the last-level inner lists
        # of the first outer list only (to avoid massive changes).
        for j in range(len(self.long_multi_level_list[0])):
            # Append person_john to each inner list in the first outer list.
            self.long_multi_level_list[0][j].append(self.test_obj)

        # New Example 2: Long multi-level dictionary
        # Create a dictionary with 10 keys at level 1; each level-1 key maps to a dictionary
        # with 1000 keys at level 2; each of these maps to another dict with 1000 key/value pairs.
        self.long_multi_level_dict = {
            f"Level1_{i}": {
                f"Level2_{j}": {f"Level3_{k}": f"Value_{i}_{j}_{k}" for k in range(1000)}
                for j in range(1000)
            }
            for i in range(10)
        }

        # For demonstration, add a person_john child to the first inner dictionary of the first outer key.
        first_outer_key = next(iter(self.long_multi_level_dict))
        first_inner_key = next(iter(self.long_multi_level_dict[first_outer_key]))
        self.long_multi_level_dict[first_outer_key][first_inner_key]["Person"] = self.test_obj

        # Large tensor (on GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.huge_tensor = torch.rand(10000, 10000).to(device)

def main():
    # Instantiate the data source
    data_source = AppDataSource()

    # Initialize the QApplication
    app = QApplication(sys.argv)

    # Instantiate the PaginatedVariableViewer with the data source.
    # This viewer includes pagination for any container level.
    viewer = VariableViewer(data_source, "c", "var_view/plugins")
    # viewer.add_console("c")  # Integrate the console
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

    periodic_timer = QTimer()
    periodic_timer.timeout.connect(log_string_var)
    periodic_timer.start(20000)  # every 20 seconds

    # Execute the application
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
