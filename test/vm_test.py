from _typeshed import Self
import sys
import pytest
# setting path
sys.path.append('../VehicleDynamicModel')

import vehiclemodel3dof


class Test_VehicleDynamicModel3dof():

    def __init__(self) -> None:
        self.vm = vehiclemodel3dof.VehicleDynamicModel3dof()

    def test_engine(self):
        self.vm.engine(200)


if __name__ == "__main__":

    pytest.main()