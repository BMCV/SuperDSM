import itertools
import signal
import multiprocessing

ipp_client = None

class Sequence:
    def __init__(self, val):
            self.val = val

def unroll(seq):
    return Sequence(seq)

def _get_args_chain(args):
    n = max(len(arg.val) for arg in list(args) if isinstance(arg, Sequence))
    return n, (arg.val if isinstance(arg, Sequence) else [arg] * n for arg in args)

class Mapper:
    VERBOSITY_SILENT   = 0
    VERBOSITY_PROGRESS = 1
    VERBOSITY_DEBUG    = 2

    def __init__(self, parallel_level=0, load_balance=True, verbosity=VERBOSITY_SILENT):
        assert ipp_client is not None, 'mapper.ipp_client must not be None'
        self.parallel_level = parallel_level
        self.level          = -1
        self.load_balance   = load_balance
        self.verbosity      = verbosity

    def map(self, f, *args):
        n, real_args = _get_args_chain(args)
        real_args = itertools.chain([f, [self] * n], real_args)
        self.level += 1
        try:
            if self.level == self.parallel_level:
                if self.verbosity >= Mapper.VERBOSITY_DEBUG:
                    print 'running %s in parallel (level: %d)' % (str(f), self.level)
                engines = ipp_client.load_balanced_view() if self.load_balance else ipp_client[:]
                async_result = engines.map(*real_args)
                if self.verbosity >= Mapper.VERBOSITY_PROGRESS:
                    async_result.wait_interactive()
                return async_result.get()
            else:
                return map(*real_args)
        finally:
            self.level -= 1

    def apply(self, f, *args):
        self.map(f, *args)

class UnrollArgs:
    def __init__(self, f):
        self.f = f
    
    def __call__(self, args):
        return self.f(*args)

class fork: # namespace

    _forked = False
    DEBUG   = False
    
    @staticmethod
    def map(processes, f, *args):
        assert processes >= 1, 'number of processes must be at least 1'
        assert not fork._forked, 'process was already forked before'

        n, real_args = _get_args_chain(args)
        real_args = list(zip(*real_args))
    
        # we need to ensure that SIGINT is handled correctly,
        # see for reference: http://stackoverflow.com/a/35134329/1444073
        if not fork.DEBUG:
            original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
            pool = multiprocessing.Pool(processes=processes)
            signal.signal(signal.SIGINT, original_sigint_handler)
        
        fork._forked = True
        try:
            if not fork.DEBUG:
                chunksize = int(round(float(n) / processes))
                result = pool.map(UnrollArgs(f), real_args, chunksize)
                pool.close()
                return result
            else:
                return map(UnrollArgs(f), real_args)
        except:
            if not fork.DEBUG: pool.terminate()
            raise
        finally:
            fork._forked = False
    
    @staticmethod
    def apply(processes, f, *args):
        fork.map(processes, f, *args)

    @staticmethod
    def imap_unordered(processes, f, *args, **kwargs):
        assert processes >= 1, 'number of processes must be at least 1'
        assert not fork._forked, 'process was already forked before'

        n, real_args = _get_args_chain(args)
        real_args = list(zip(*real_args))
    
        if not fork.DEBUG: pool = multiprocessing.Pool(processes=processes)
        fork._forked = True
        try:
            if not fork.DEBUG:
                chunksize = int(round(float(n) / processes)) if kwargs.get('use_chunks') else 1
                for result in pool.imap_unordered(UnrollArgs(f), real_args, chunksize):
                    yield result
                pool.close()
            else:
                for result in itertools.imap(UnrollArgs(f), real_args):
                    yield result
        except:
            if not fork.DEBUG: pool.terminate()
            raise
        finally:
            fork._forked = False

