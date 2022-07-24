import regemo.config as config

import os
import shutil

def create_dir(path,
               delete=False):
    """
    Parameters
    ----------
    path: path to the final folder
    delete: whether you want to delete folders if they exist

    Returns
    -------
    NULL
    """
    def _rec_split(s):
        rest, tail = os.path.split(s)
        if rest == "" or rest == "/":
            return [tail]

        return _rec_split(rest) + [tail]

    def _make_dir(dir_path,
                  delete=False):


        if os.path.isdir(dir_path):
            if delete:
                shutil.rmtree(dir_path)
                os.mkdir(dir_path)
        else:
            os.mkdir(dir_path)

    path_dirs = _rec_split(path)
    cur_path = config.BASE_PATH

    for subdir in path_dirs[:-1]:
        # TODO: add a check for files
        cur_path = os.path.join(cur_path, subdir)
        if os.path.isfile(cur_path):
            return

        _make_dir(cur_path)

    _make_dir(os.path.join(cur_path, path_dirs[-1]), delete=delete)

def helper_file_naming(name):
    # some of the characters in the sensor naming might not be appropriate for files
    special_chars = "#<>$%{}/+!`&*'?=/\:@"  # characters not allowed as part of the naming
    modified_name = name
    for c in special_chars:
        modified_name = modified_name.replace(c, '')
    return modified_name

def main():
    create_dir(f"model_storage/yield_prediction/result-yo")

if __name__ == "__main__":
    main()