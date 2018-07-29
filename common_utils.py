import errno
import os


def open_path_or_create(path_dir):
    if not os.path.exists(os.path.dirname(path_dir)):
        try:
            os.makedirs(os.path.dirname(path_dir))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
