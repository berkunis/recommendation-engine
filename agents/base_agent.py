from abc import ABC, abstractmethod
from utils.printing import print_section_header


class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name

    def print_header(self):
        print_section_header(f"Agent: {self.name}")

    @abstractmethod
    def run(self, **kwargs) -> dict:
        pass
