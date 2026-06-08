"""
EATA消融实验变体模块
包含5个消融变体的实现（不包含EATA-Full主实验）
"""

from .eata_nonn import EATANoNN
from .eata_nomem import EATANoMem
from .eata_simple import EATASimple
from .eata_noexplore import EATANoExplore
from .eata_nomcts import EATANoMCTS

__all__ = [
    'EATANoNN', 
    'EATANoMem',
    'EATASimple',
    'EATANoExplore',
    'EATANoMCTS'
]
