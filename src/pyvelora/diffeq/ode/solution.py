from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Solution:
    t: list
    y: list

    def final(self):
        return self.y[-1]

    def plot(self):
        from .plotting.solution_plot import solution_plot
        solution_plot(self)

    def phase(self):
        from .plotting.phase_plot import phase_plot
        phase_plot(self)