import baseline


class ArgumentParser(baseline.ArgumentParser):
    def __init__(self, description='baseline', **kwargs):
        super(ArgumentParser, self).__init__(description=description)

    def add_arguments(self):
        super().add_arguments()

        self.add_argument('--sparse', default=False, action='store_true')
        self.add_argument('--sparse_activation', type=str, default='relu')
        self.add_argument('--no_dense', default=False, action='store_true')

        self.add_argument('--dual', default=False, action='store_true')
        self.add_argument('--dual_init', type=float, default=5.0)
        self.add_argument('--dual_hl', type=float, default=10000)

        self.add_argument('--phrase_filter', default=False, action='store_true')
        self.add_argument('--filter_init', type=float, default=0.1)
        self.add_argument('--filter_th', type=float, default=0.0)

        self.add_argument('--multimodal', default=False, action='store_true')

    def parse_args(self, **kwargs):
        args = super().parse_args()

        args.dense = not args.no_dense
        if args.sparse:
            args.emb_type = 'sparse'

        return args
