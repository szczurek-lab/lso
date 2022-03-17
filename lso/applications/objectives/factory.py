from lso.applications.objectives import mnist as lso_mnist_obj


class ObjectiveFactory:

    @classmethod
    def get_from_args(cls, args):
        if args['name'] == 'MNISTSumObjectiveFunction':
            return lso_mnist_obj.MNISTSumObjectiveFunction()
        raise ValueError(f'Invalid args provided do ObjectiveFactory {args}.')
