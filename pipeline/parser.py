import pandas as pd
import numpy as np
from typing import Path, List
import os


class _Base:
    def __init__(self, raw_text: Path):
        self._raw_text = raw_text
        self._title = self._extract_title()

    def _extract_title(self) -> str:
        return os.path.basename(self.raw_text)


class LatinVulgate(_Base):
    def __init__(self, raw_text: Path):
        super().__init__(raw_text)

    def _extract_chapter_titles(self) -> List:
        pass
