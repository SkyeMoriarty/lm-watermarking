from abc import ABC, abstractmethod


class Attacker(ABC):
    @abstractmethod
    def attack(self, text, device, epsilon):
        pass
