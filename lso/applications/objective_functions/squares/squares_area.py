import numpy as np
from typing import Optional

from lso.utils import area_utils
from lso.data import data as lso_data
from lso.objective_function import objective_function as lso_objective_function


class SquaresArea(lso_objective_function.ObjectiveFunction):

    def __init__(self, shape_likeliness_coeff: Optional[float] = 1.0):
        self.shape_likeliness_coeff = shape_likeliness_coeff

    def calculate_shape_likeliness(self,
                                   x: np.array,
                                   metric: Optional[str] = 'Jaccard') -> np.array:

        if metric == 'Jaccard':

            # Compute the Jaccard distance for areas of: shape x ,and the square circumscribed on shape x.

            area_x = area_utils.calculate_area(x)
            area_circumscribed_x = area_utils.calculate_circumscribed_square_area(x)

            jaccard_index = area_x / area_circumscribed_x
            shape_likeliness = 1 - jaccard_index

        if metric not in ['Jaccard']:
            raise ValueError('provide a correct metric.')

        return shape_likeliness

    def evaluate(self, data: lso_data.Data) -> lso_data.Data:

        assert issubclass(type(data), lso_objective_function.ObjectiveFunction)
        assert len(data.x.shape) == 3

        area = area_utils.calculate_area(data.x)
        shape_likeliness = self.shape_likeliness_coeff * self.calculate_shape_likeliness(data.x)
        objective = area - shape_likeliness

        return type(data)(
            x=data.x,
            objective=objective,
            features=data.features
        )
