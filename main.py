import os

import tests as t

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Supress warning and informational messages


def main():
    t.test()
    exit(0)


if __name__ == '__main__':
    main()
