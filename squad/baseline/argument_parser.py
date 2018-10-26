import os
import base


class ArgumentParser(base.ArgumentParser):
    def __init__(self, description='baseline', **kwargs):
        super(ArgumentParser, self).__init__(description=description)

    def add_arguments(self):
        super().add_arguments()

        home = os.path.expanduser('~')

        # Metadata paths
        self.add_argument('--static_dir', type=str, default=os.path.join(home, 'data'))
        self.add_argument('--glove_dir', type=str, default=None, help='location of GloVe')
        self.add_argument('--elmo_options_file', type=str, default=None)
        self.add_argument('--elmo_weights_file', type=str, default=None)

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

        # Training arguments. Only valid during training
        self.add_argument('--dropout', type=float, default=0.2)
        self.add_argument('--max_context_size', type=int, default=256)
        self.add_argument('--max_question_size', type=int, default=32)
        self.add_argument('--no_bucket', default=False, action='store_true')
        self.add_argument('--no_shuffle', default=False, action='store_true')

        # Other arguments
        self.add_argument('--emb_type', type=str, default='dense', help='dense|sparse')
        self.add_argument('--glove_cuda', default=False, action='store_true')

    def parse_args(self, **kwargs):
        args = super().parse_args()

        if args.draft:
            args.glove_vocab_size = 102

        if args.glove_dir is None:
            args.glove_dir = os.path.join(args.static_dir, 'glove')
        if args.elmo_options_file is None:
            args.elmo_options_file = os.path.join(args.static_dir, 'elmo', 'options.json')
        if args.elmo_weights_file is None:
            args.elmo_weights_file = os.path.join(args.static_dir, 'elmo', 'weights.json')

        args.embed_size = args.glove_size
        args.glove_cpu = not args.glove_cuda
        args.bucket = not args.no_bucket
        args.shuffle = not args.no_shuffle
        return args
