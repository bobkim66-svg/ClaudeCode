"""PyInstaller / 직접 실행용 진입 스크립트: python run_polarsim.py"""

import sys

from polarsim.app import main

if __name__ == "__main__":
    sys.exit(main())
