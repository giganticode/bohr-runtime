from bohrruntime import setup_loggers
from bohrruntime.config.appconfig import AppConfig
from bohrruntime.config.pathconfig import add_to_local_config


class verbosity:
    def __init__(self, verbose: bool = True):
        self.current_verbosity = AppConfig.load().verbose
        self.verbose = verbose or self.current_verbosity

    def __enter__(self):
        add_to_local_config("core.verbose", str(self.verbose))
        setup_loggers(self.verbose)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        add_to_local_config("core.verbose", str(self.current_verbosity))
