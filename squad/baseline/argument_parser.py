import os
import base.argument_parser


class ArgumentParser(base.argument_parser.ArgumentParser):
    def __init__(self, description='baseline', **kwargs):
        super(ArgumentParser, self).__init__(description=description)

    def add_arguments(self):
        super().add_arguments()

        home = os.path.expanduser('~')

        # Metadata paths
        self.add_argument('--glove_dir', type=str, default=os.path.join(home, 'data', 'glove'),
                          help='location of GloVe')
        self.add_argument('--elmo_options_file', type=str, default=os.path.join(home, 'data', 'elmo', 'options.json'))
        self.add_argument('--elmo_weights_file', type=str, default=os.path.join(home, 'data', 'elmo', 'weights.hdf5'))

        # Model arguments
        self.add_argument('--word_vocab_size', type=int, default=10000)
        self.add_argument('--char_vocab_size', type=int, default=100)
        self.add_argument('--glove_vocab_size', type=int, default=400002)
        self.add_argument('--glove_size', type=int, default=200)
        self.add_argument('--hidden_size', type=int, default=128)
        self.add_argument('--batch_size', type=int, default=64, help='batch size')
        self.add_argument('--elmo', default=False, action='store_true')
        self.add_argument('--num_heads', type=int, default=1)
        self.add_argument('--max_pool', default=False, action='store_true')
        self.add_argument('--agg', type=str, default='max', help='max|logsumexp')
        self.add_argument('--num_layers', type=int, default=1)

        # Training arguments
        self.add_argument('--dropout', type=float, default=0.2)

        # Other arguments
        self.add_argument('--emb_type', type=str, default='dense')
        self.add_argument('--glove_cuda', default=False, action='store_true')

    def parse_args(self, **kwargs):
        args = super().parse_args()
        args.embed_size = args.glove_size
        args.glove_cpu = not args.glove_cuda
        return args
