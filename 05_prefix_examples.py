def train_fn(root, prefix=None):
    from ml_logger import logger

    logger.configure(root_dir=root, prefix=prefix)
    logger.log_line(prefix, "launch is working.", file="o_hey")


if __name__ == '__main__':
    import os
    import jaynes
    from ml_logger import logger, ROOT

    jaynes.config('local')
    jaynes.run(train_fn, root=ROOT, prefix=os.getcwd())

    jaynes.listen()
