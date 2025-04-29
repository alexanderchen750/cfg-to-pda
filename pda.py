from typing import Dict, List, Set, Tuple, Optional, Union, Literal, FrozenSet
from pdaSymbol import PDASymbol
from cfg import ContextFreeGrammar


class PDAStack:
    """Stack for PDA operations."""
    def __init__(self, initial_symbol: Optional[PDASymbol] = None):
        self.items = [initial_symbol] if initial_symbol else []
    
    def push(self, symbol: PDASymbol) -> None:
        """Push symbol onto stack."""
        self.items.append(symbol)
    
    def pop(self) -> PDASymbol:
        """Pop and return top symbol from stack."""
        if not self.items:
            raise ValueError("Cannot pop from empty stack")
        return self.items.pop()
    
    def top(self) -> Optional[PDASymbol]:
        """Return top symbol without popping."""
        if not self.items:
            return None
        return self.items[-1]
    
    def replace(self, symbols: Union[PDASymbol, List[PDASymbol]], in_place = True) -> 'PDAStack':
        """Replace top symbol with one or more symbols. Pushing in order formultiple symbols"""
        if in_place:
            if self.items:  
                self.items.pop()
                
                if isinstance(symbols, PDASymbol):
                    self.push(symbols)
                else:
                    for symbol in symbols: 
                        self.push(symbol)
                
                return self
            else:
                raise ValueError("Cannot replace symbol in empty stack")
        else:
            new_stack = PDAStack()
            new_stack.items = self.items[:-1].copy()
            if isinstance(symbols, PDASymbol):
                new_stack.push(symbols)
            else:
                for symbol in symbols:
                    new_stack.push(symbol)
            
            return new_stack
    
    def is_empty(self) -> bool:
        """Check if stack is empty."""
        return len(self.items) == 0
    
    def copy(self) -> 'PDAStack':
        """Create a copy of this stack."""
        new_stack = PDAStack()
        new_stack.items = self.items.copy()
        return new_stack
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __str__(self) -> str:
        return str([str(item) for item in self.items])

class PDAConfiguration:
    """Configuration of a PDA at a given point in computation."""
    def __init__(self, state: str, remaining_input: List[PDASymbol], stack: PDAStack):
        self.state = state
        self.remaining_input = remaining_input
        self.stack = stack
    
    def __str__(self) -> str:
        input_str = ''.join(str(s) for s in self.remaining_input)
        return f"({self.state}, {input_str}, {self.stack})"

class PDA:
    """Pushdown automaton implementation."""
    
    def __init__(self, 
                 states: Set[str],
                 input_symbols: Set[PDASymbol],
                 stack_symbols: Set[PDASymbol],
                 transitions: Dict[str, Dict[Optional[PDASymbol], Dict[PDASymbol, List[Tuple[str, List[PDASymbol]]]]]],
                 initial_state: str,
                 initial_stack_symbol: PDASymbol,
                 final_states: Set[str],
                 acceptance_mode: Literal["final_state", "empty_stack", "both"] = "final_state"):
        """
        Initialize a pushdown automaton.
        
        Args:
            states: Set of states
            input_symbols: Set of input symbols
            stack_symbols: Set of stack symbols
            transitions: Nested dictionary of transitions:
                         {state: {input_symbol: {stack_symbol: [(next_state, [new_stack_symbols])]}}
            initial_state: Starting state
            initial_stack_symbol: Initial stack symbol
            final_states: Set of accepting states
            acceptance_mode: "final_state", "empty_stack", or "both"

            for transtions:
                transitions[current_state][input_symbol][stack_top] = [
                    (next_state, [new_stack_symbols])
                ]
        """
        self.states = states
        self.input_symbols = input_symbols
        self.stack_symbols = stack_symbols
        self.transitions = transitions
        self.initial_state = initial_state
        self.initial_stack_symbol = initial_stack_symbol
        self.final_states = final_states
        self.acceptance_mode = acceptance_mode
        
        # Validate the PDA
        self.validate()
    
    def validate(self):
        """Check that the PDA is properly defined."""
        # Check states
        if self.initial_state not in self.states:
            raise ValueError(f"Initial state {self.initial_state} is not in the set of states")
        
        for state in self.final_states:
            if state not in self.states:
                raise ValueError(f"Final state {state} is not in the set of states")
        
        # Check initial stack symbol
        if self.initial_stack_symbol not in self.stack_symbols:
            raise ValueError(f"Initial stack symbol {self.initial_stack_symbol} is not valid")
        
        # Check transitions
        for state, state_transitions in self.transitions.items():
            if state not in self.states:
                raise ValueError(f"State {state} in transitions is not in the set of states")
            
            for input_symbol, input_transitions in state_transitions.items():
                if input_symbol is not None and input_symbol not in self.input_symbols:
                    raise ValueError(f"Input symbol {input_symbol} in transitions is not valid")
                
                for stack_symbol, stack_transitions in input_transitions.items():
                    if stack_symbol not in self.stack_symbols:
                        raise ValueError(f"Stack symbol {stack_symbol} in transitions is not valid")
                    
                    for next_state, new_stack_symbols in stack_transitions:
                        if next_state not in self.states:
                            raise ValueError(f"Next state {next_state} in transitions is not valid")
                        
                        for symbol in new_stack_symbols:
                            if symbol not in self.stack_symbols:
                                raise ValueError(f"New stack symbol {symbol} in transitions is not valid")
    
    def accepts_input(self, input_str: str) -> bool:
        """Check if the PDA accepts the given input string."""
        # Convert string to symbols
        input_symbols = [PDASymbol(c) for c in input_str]
        paths, is_accepted = self._get_input_path(input_symbols)
        return is_accepted
    
    def _get_input_path(self, input_symbols: List[PDASymbol]) -> Tuple[List[Tuple[PDAConfiguration, PDAConfiguration]], bool]:
        """The ENGINE: Calculate the path taken by input.
            Peform BFS(with get_next_configuartions) to find a path to acceptance.
            Returns a list of tuples (current_config, next_config) and a boolean indicating acceptance.
            Mark visited configurations to avoid cycles.
        """
        # initial configuration
        stack = PDAStack(self.initial_stack_symbol)
        initial_config = PDAConfiguration(self.initial_state, input_symbols, stack)
        
        # breadth-first search to find a path to acceptance
        visited = set()
        queue = [(initial_config, [])]
        
        while queue:
            current_config, path = queue.pop(0)
            
            # Create a hashable representation of the configuration
            stack_tuple = tuple((s.value, s.is_terminal) for s in current_config.stack.items)
            input_tuple = tuple((s.value, s.is_terminal) for s in current_config.remaining_input)
            config_key = (current_config.state, input_tuple, stack_tuple)
            
            # Skip if we've seen this configuration
            if config_key in visited:
                continue
            
            visited.add(config_key)
            
            # Check if the configuration accepts
            if self._has_accepted(current_config):
                return path, True
            
            # Add all possible next configurations
            for next_config in self._get_next_configurations(current_config):
                new_path = path + [(current_config, next_config)]
                queue.append((next_config, new_path))
        
        # If we've exhausted all possibilities without finding an accepting configuration
        return [], False
    
    def _get_next_configurations(self, config: PDAConfiguration) -> List[PDAConfiguration]:
        """Get all possible next configurations from the current one."""
        result = []
        
        # Get the stack top
        stack_top = config.stack.top()
        if stack_top is None:
            return result
        
        # If there are lambda transitions, add them
        if self._has_lambda_transition(config.state, stack_top):
            for next_state, new_stack_symbols in self.transitions[config.state][None][stack_top]:
                new_stack = config.stack.copy()
                new_stack.pop()  # Pop the matched symbol
                
                # Push new symbols in reverse order (last in, first out)
                for symbol in reversed(new_stack_symbols):
                    new_stack.push(symbol)
                
                new_config = PDAConfiguration(next_state, config.remaining_input, new_stack)
                result.append(new_config)
        
        # If there are transitions for the current input symbol, add them
        if config.remaining_input and config.state in self.transitions:
            input_symbol = config.remaining_input[0]
            
            # Check if there's a transition for this input symbol
            if input_symbol in self.transitions.get(config.state, {}):
                if stack_top in self.transitions[config.state][input_symbol]:
                    for next_state, new_stack_symbols in self.transitions[config.state][input_symbol][stack_top]:
                        new_stack = config.stack.copy()
                        new_stack.pop()  # Pop the matched symbol
                        
                        # Push new symbols in reverse order
                        for symbol in reversed(new_stack_symbols):
                            new_stack.push(symbol)
                        
                        new_config = PDAConfiguration(
                            next_state, 
                            config.remaining_input[1:],  # Consume the input symbol
                            new_stack
                        )
                        result.append(new_config)
        
        return result
    
    def _has_lambda_transition(self, state: str, stack_symbol: PDASymbol) -> bool:
        """Check if there's a lambda transition from the current state with the given stack top."""
        return (
            state in self.transitions
            and None in self.transitions[state]
            and stack_symbol in self.transitions[state][None]
        )
    
    def _has_accepted(self, config: PDAConfiguration) -> bool:
        """Check if the configuration indicates an accepted input."""
        # If there's input left, we're not accepted
        if config.remaining_input:
            return False
        
        # Check acceptance mode
        if self.acceptance_mode == "both":
            return config.state in self.final_states and config.stack.is_empty()
        elif self.acceptance_mode == "empty_stack":
            return config.stack.is_empty()
        elif self.acceptance_mode == "final_state":
            return config.state in self.final_states

        
        return False
    
    def run_with_branching(self, input_str: str, max_branches: int = 10, return_only_final: bool = False) -> List[PDAConfiguration]:
        """
        Run the PDA on input with limited branching for exploration.
        Returns a list of configurations that could be reached.

        This is a breadth-first search (BFS) with a limit on the number of branches
        Like _get_input_path, but with a limit on the number of branches at each step and for all configurations instead of just the path.
        
        Args:
            input_str: Input string to process
            max_branches: Maximum number of branches to explore at each step
            return_only_final: If True, return only accepting configurations

        Returns:
            List of PDAConfiguration objects representing the configurations reached
        """
        # Convert string to symbols
        input_symbols = [PDASymbol(c) for c in input_str]
        
        # Start with initial configuration
        stack = PDAStack(self.initial_stack_symbol)
        initial_config = PDAConfiguration(self.initial_state, input_symbols, stack)
        
        # BFS with limited branching
        visited = set()
        queue = [initial_config]
        
        # Initialize all_configs based on return_only_final flag
        all_configs = [] if return_only_final else [initial_config]
        final_configs = []
        
        while queue:
            current_config = queue.pop(0)
            
            # Create a hashable representation of the configuration
            stack_tuple = tuple((s.value, s.is_terminal) for s in current_config.stack.items)
            input_tuple = tuple((s.value, s.is_terminal) for s in current_config.remaining_input)
            config_key = (current_config.state, input_tuple, stack_tuple)
            
            # Skip if we've seen this configuration
            if config_key in visited:
                continue
            
            visited.add(config_key)
            
            # Check if this is an accepting configuration
            if self._has_accepted(current_config):
                final_configs.append(current_config)
                if return_only_final:
                    all_configs.append(current_config)
            
            # Check if this is a dead configuration (no input left and not accepting)
            if not current_config.remaining_input and not self._has_accepted(current_config):
                # Skip adding next configurations for dead ends
                continue
            
            # Get next possible configurations
            next_configs = self._get_next_configurations(current_config)
            
            # Limit branching if needed
            if len(next_configs) > max_branches:
                next_configs = next_configs[:max_branches]
            
            # Add to queue and all_configs
            for config in next_configs:
                queue.append(config)
                if not return_only_final:
                    all_configs.append(config)
        
        # Return final_configs if return_only_final is True, otherwise all_configs
        return all_configs
    
    def generate_weighted_tuples(self, input_samples: List[str], top_k: int = 5) -> List[Tuple[Tuple[Tuple[Tuple[str, bool], ...], str], float]]:
        """
        Generate (stack_state, next_token) tuples with weights for training based on corpus data.
        
        Args:
            input_samples: List of input sequences
            top_k: Number of stack symbols to consider
                
        Returns:
            List of ((stack_state, next_token), weight) tuples
        """
        tuple_counts = {}
        total_count = 0
        
        for input_seq in input_samples:
            # Convert string to symbols
            input_symbols = [PDASymbol(c) for c in input_seq]
            
            # Start with initial configuration
            stack = PDAStack(self.initial_stack_symbol)
            config = PDAConfiguration(self.initial_state, input_symbols, stack)
            
            # Process input deterministically, following first valid path
            while config.remaining_input:
                # Extract stack state (top k symbols)
                stack_items = config.stack.items
                stack_state = tuple(((s.value, s.is_terminal) for s in stack_items[-top_k:]) 
                                if len(stack_items) >= top_k else 
                                ((s.value, s.is_terminal) for s in stack_items))
                
                # Record the current symbol and stack state
                next_symbol = config.remaining_input[0].value
                tuple_key = (stack_state, next_symbol)
                tuple_counts[tuple_key] = tuple_counts.get(tuple_key, 0) + 1
                total_count += 1
                
                # Get next configurations and choose the first valid one
                next_configs = self._get_next_configurations(config)
                if not next_configs:
                    break  # No valid next configuration
                
                # In a real implementation, you might want to score these and choose the best
                config = next_configs[0]  # Take first available path
        
        # Convert to weighted tuples
        weighted_tuples = []
        for tuple_key, count in tuple_counts.items():
            if total_count > 0:
                weight = count / total_count
                weighted_tuples.append((tuple_key, weight))
        
        return weighted_tuples


def convert_grammar_to_pda(grammar: ContextFreeGrammar) -> PDA:
    """
    Convert a context-free grammar to a PDA.
    This implements the standard algorithm for converting CFG to PDA.
    """
    # First, expand any EBNF constructs to standard CFG
    expanded_grammar = grammar.expand_ebnf()
    
    # Create states
    states = {'q0', 'q1', 'q2'}
    
    # Create bottom of stack symbol
    bottom_symbol = PDASymbol('$', True)
    
    # Get terminals and non-terminals
    input_symbols = expanded_grammar.terminals
    stack_symbols = expanded_grammar.terminals.union(expanded_grammar.non_terminals).union({bottom_symbol})
    
    # Create transitions dictionary
    transitions = {}
    
    # Initialize transitions for each state
    for state in states:
        transitions[state] = {}
        transitions[state][None] = {}  # For epsilon transitions
    
    # Add transition from q0 to q1, pushing start symbol on top of $
    transitions['q0'][None][bottom_symbol] = [('q1', [expanded_grammar.start_symbol, bottom_symbol])]
    
    # For each non-terminal, add epsilon transitions to replace with RHS
    for production in expanded_grammar.productions:
        lhs = production.lhs
        rhs = production.rhs
        
        if lhs not in transitions['q1'][None]:
            transitions['q1'][None][lhs] = []
        
        if not rhs:  # Epsilon production
            transitions['q1'][None][lhs].append(('q1', []))
        else:
            transitions['q1'][None][lhs].append(('q1', rhs))
    
    # For each terminal, add transition to match and pop
    for terminal in expanded_grammar.terminals:
        if terminal not in transitions['q1']:
            transitions['q1'][terminal] = {}
        
        transitions['q1'][terminal][terminal] = [('q1', [])]
    
    # Add transition to final state when only $ remains on stack
    transitions['q1'][None][bottom_symbol] = [('q2', [bottom_symbol])]
    
    # Create and return the PDA
    return PDA(
        states=states,
        input_symbols=input_symbols,
        stack_symbols=stack_symbols,
        transitions=transitions,
        initial_state='q0',
        initial_stack_symbol=bottom_symbol,
        final_states={'q2'},
        acceptance_mode="final_state"
    )
# Example usage
def example():
    # Create a simple grammar
    grammar = ContextFreeGrammar(
        non_terminals={'S', 'A', 'B'},
        terminals={'a', 'b', 'c'},
        productions=[
            ('S', ['A', 'B']),
            ('A', ['a', 'A']),
            ('A', []),  # Epsilon
            ('B', ['b', 'B']),
            ('B', ['c'])
        ],
        start_symbol='S'
    )
    
    # Convert grammar to PDA
    pda = convert_grammar_to_pda(grammar)
    
    # Test some inputs
    test_inputs = ['abc', 'cac', 'abbc', 'aaabbc']
    for input_str in test_inputs:
        is_accepted = pda.accepts_input(input_str)
        print(f"Input '{input_str}' is {'accepted' if is_accepted else 'rejected'}")
    
    # Generate weighted tuples for training
    weighted_tuples = pda.generate_weighted_tuples(test_inputs, top_k=3)
    print("Weighted tuples for training:")
    for (stack_state, next_symbol), weight in weighted_tuples:
        # Convert the tuple representation to a more readable format
        stack_str = [(val, "T" if is_term else "NT") for val, is_term in stack_state]
        print(f"(({stack_str}, {next_symbol}), {weight:.3f})")
    
    return pda

if __name__ == "__main__":
    example()