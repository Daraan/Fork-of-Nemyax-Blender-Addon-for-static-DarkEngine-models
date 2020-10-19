from pathlib import Path
import sys

p = Path(__file__)
sys.path.insert(0, str(p.parent.resolve()))

from ._gif2png import convert as convert

