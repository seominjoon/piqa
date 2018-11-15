import argparse
import os


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, description='base', **kwargs):
        super(ArgumentParser, self).__init__(description=description)

    def add_arguments(self):
        home = os.path.expanduser('~')
        self.add_argument('model', type=str)

        self.add_argument('--mode', type=str, default='train')
        self.add_argument('--iteration', type=str, default='1')
        self.add_argument('--pause', type=int, default=0)  # ignore this argument.

        # Data (input) paths
        self.add_argument('--train_path', type=str, default=os.path.join(home, 'data', 'squad', 'train-v1.1.json'),
                          help='location of the training data')
        self.add_argument('--test_path', type=str, default=os.path.join(home, 'data', 'squad', 'dev-v1.1.json'),
                          help='location of the test data')

        # Output paths
        self.add_argument('--output_dir', type=str, default='/tmp/piqa/squad/', help='Output directory')
        self.add_argument('--save_dir', type=str, default=None, help='location for saving the model')
        self.add_argument('--load_dir', type=str, default=None, help='location for loading the model')
        self.add_argument('--dump_dir', type=str, default=None, help='location for dumping outputs')
        self.add_argument('--report_path', type=str, default=None, help='location for report')
        self.add_argument('--pred_path', type=str, default=None, help='location for prediction json file during `test`')
        self.add_argument('--cache_path', type=str, default=None)
        self.add_argument('--question_emb_dir', type=str, default=None)
        self.add_argument('--context_emb_dir', type=str, default=None)

        # Training arguments
        self.add_argument('--epochs', type=int, default=20)
        self.add_argument('--train_steps', type=int, default=0)
        self.add_argument('--eval_steps', type=int, default=1000)
        self.add_argument('--eval_save_period', type=int, default=500)
        self.add_argument('--report_period', type=int, default=100)

        # Similarity search (faiss, pysparnn) arguments
        self.add_argument('--metric', type=str, default='ip', help='ip|l2')
        self.add_argument('--nlist', type=int, default=1)
        self.add_argument('--nprobe', type=int, default=1)
        self.add_argument('--bpv', type=int, default=None, help='bytes per vector (e.g. 8)')
        self.add_argument('--num_train_mats', type=int, default=100)

        # Demo arguments
        self.add_argument('--port', type=int, default=8080)

        # Other arguments
        self.add_argument('--draft', default=False, action='store_true')
        self.add_argument('--cuda', default=False, action='store_true')
        self.add_argument('--preload', default=False, action='store_true')
        self.add_argument('--cache', default=False, action='store_true')
        self.add_argument('--archive', default=False, action='store_true')
        self.add_argument('--dump_period', type=int, default=20)
        self.add_argument('--emb_type', type=str, default='dense', help='dense|sparse')
        self.add_argument('--metadata', default=False, action='store_true')
        self.add_argument('--mem_info', default=False, action='store_true')

    def parse_args(self, **kwargs):
        args = super().parse_args()
        if args.draft:
            args.batch_size = 2
            args.eval_steps = 1
            args.eval_save_period = 2
            args.train_steps = 2

        if args.save_dir is None:
            args.save_dir = os.path.join(args.output_dir, 'save')
        if args.load_dir is None:
            args.load_dir = os.path.join(args.output_dir, 'save')
        if args.dump_dir is None:
            args.dump_dir = os.path.join(args.output_dir, 'dump')
        if args.question_emb_dir is None:
            args.question_emb_dir = os.path.join(args.output_dir, 'question_emb')
        if args.context_emb_dir is None:
            args.context_emb_dir = os.path.join(args.output_dir, 'context_emb')
        if args.report_path is None:
            args.report_path = os.path.join(args.output_dir, 'report.csv')
        if args.pred_path is None:
            args.pred_path = os.path.join(args.output_dir, 'pred.json')
        if args.cache_path is None:
            args.cache_path = os.path.join(args.output_dir, 'cache.b')

        args.load_dir = os.path.abspath(args.load_dir)
        args.context_emb_dir = os.path.abspath(args.context_emb_dir)
        args.question_emb_dir = os.path.abspath(args.question_emb_dir)

        return args
