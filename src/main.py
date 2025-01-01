# src/main.py
import sys
import logging
import numpy as np
import torch
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer, QObject, pyqtSignal

from variable_viewer import VariableViewer  # Ensure this module is correctly implemented
from variables import Variables  # Ensure this module is correctly implemented


# Configure logging for debugging purposes
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to see detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# Define your custom classes with signals
class Engine(QObject):
    horsepower_changed = pyqtSignal(int)
    type_changed = pyqtSignal(str)

    def __init__(self, horsepower, type_):
        super().__init__()
        self._horsepower = horsepower
        self._type = type_

    @property
    def horsepower(self):
        return self._horsepower

    @horsepower.setter
    def horsepower(self, value):
        if self._horsepower != value:
            self._horsepower = value
            self.horsepower_changed.emit(value)

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        if self._type != value:
            self._type = value
            self.type_changed.emit(value)


class Car(QObject):
    make_changed = pyqtSignal(str)
    model_changed = pyqtSignal(str)
    engine_changed = pyqtSignal(QObject)
    owner_changed = pyqtSignal(QObject)

    def __init__(self, make, model, engine):
        super().__init__()
        self._make = make
        self._model = model
        self._engine = engine
        self._owner = None  # To be set later, creating a cyclic reference

        # Connect engine signals to propagate changes
        if isinstance(engine, Engine):
            engine.horsepower_changed.connect(lambda val: self.engine_changed.emit(self._engine))
            engine.type_changed.connect(lambda val: self.engine_changed.emit(self._engine))

    @property
    def make(self):
        return self._make

    @make.setter
    def make(self, value):
        if self._make != value:
            self._make = value
            self.make_changed.emit(value)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        if self._model != value:
            self._model = value
            self.model_changed.emit(value)

    @property
    def engine(self):
        return self._engine

    @engine.setter
    def engine(self, value):
        if self._engine != value:
            self._engine = value
            self.engine_changed.emit(value)

    @property
    def owner(self):
        return self._owner

    @owner.setter
    def owner(self, value):
        if self._owner != value:
            self._owner = value
            self.owner_changed.emit(value)

    def set_owner(self, owner):
        """
        Set the owner of the car and establish a cyclic reference.
        """
        self.owner = owner
        logging.info(f"Car owner set to {owner.name if owner else 'None'}")


class Person(QObject):
    name_changed = pyqtSignal(str)
    age_changed = pyqtSignal(int)
    car_changed = pyqtSignal(QObject)

    def __init__(self, name, age, car=None):
        super().__init__()
        self._name = name
        self._age = age
        self._car = car

        # Connect car signals to propagate changes
        if isinstance(car, Car):
            car.make_changed.connect(lambda val: self.car_changed.emit(self._car))
            car.model_changed.connect(lambda val: self.car_changed.emit(self._car))
            car.engine_changed.connect(lambda val: self.car_changed.emit(self._car))
            car.owner_changed.connect(lambda val: self.car_changed.emit(self._car))

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if self._name != value:
            self._name = value
            self.name_changed.emit(value)

    @property
    def age(self):
        return self._age

    @age.setter
    def age(self, value):
        if self._age != value:
            self._age = value
            self.age_changed.emit(value)

    @property
    def car(self):
        return self._car

    @car.setter
    def car(self, value):
        if self._car != value:
            self._car = value
            self.car_changed.emit(value)

    def buy_car(self, car):
        """
        Purchase a car and set the owner of the car to self.
        """
        self.car = car
        car.set_owner(self)
        logging.info(f"{self.name} bought a {car.make} {car.model}")

    def greet(self):
        return f"Hello, my name is {self.name}."


def main():
    # Instantiate the Variables class
    app_variables = Variables()

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
    app_variables.set_variable("torch_tensor", torch.rand(10, 10).to(device))
    app_variables.set_variable("string_var", "Try right click update!")
    app_variables.set_variable("test_obj", person_john)  # Custom object
    app_variables.set_variable("list_obj", [person_john, person_john, person_john])
    app_variables.set_variable("huge_tensor", torch.rand(10000, 10000))
    app_variables.set_variable("complex_nested_dict", {
        "level1": {
            "level2": {
                "level3": [1, 2, 3, 4, {"deep": "value"}]
            }
        }
    })
    app_variables.set_variable("cyclic_ref", {})
    app_variables.cyclic_ref["self"] = app_variables.cyclic_ref  # Establish cyclic reference

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
