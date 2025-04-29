
class PDASymbol:
    """Symbol that can appear in input or on stack."""
    def __init__(self, value: str, is_terminal: bool = True):
        self.value = value
        self.is_terminal = is_terminal
    
    def __eq__(self, other):
        if isinstance(other, PDASymbol):
            return self.value == other.value and self.is_terminal == other.is_terminal
        return False
    
    def __hash__(self):
        return hash((self.value, self.is_terminal))
    
    def __str__(self):
        return self.value
    
    def __repr__(self):
        return f"PDASymbol({self.value}, {self.is_terminal})"