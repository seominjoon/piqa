import dev


class ArgumentParser(dev.ArgumentParser):
    def __init__(self, description='baseline', **kwargs):
        super(ArgumentParser, self).__init__(description=description)

    def add_arguments(self):
        super().add_arguments()

        self.add_argument('--max_eval_par', type=int, default=0)

    def parse_args(self, **kwargs):
        args = super().parse_args()

        return args
