"""
Run a CoinRun environment in a window where you can interact with it using the keyboard
"""

from coinrun.coinrunenv import lib
from coinrun import setup_utils


def main():
    setup_utils.setup_and_load(paint_vel_info=0)
    print("""Control with arrow keys,
F1, F2 -- switch resolution,
F5, F6, F7, F8 -- zoom,
F9  -- switch reconstruction target picture,
F10 -- switch lasers
    """)
    lib.test_main_loop()


if __name__ == '__main__':
    main()