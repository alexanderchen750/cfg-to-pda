from typing import List, Set, Tuple
from pdaSymbol import PDASymbol
from typing import Dict, List, Set, Tuple, Optional, Union, Literal, FrozenSet

class EBNFConstruct:
    """Represents EBNF constructs like optional (?), repetition (*), and positive closure (+)."""
    
    OPTIONAL = '?'      # Zero or one occurrence
    STAR = '*'          # Zero or more occurrences
    PLUS = '+'          # One or more occurrences
    GROUP = 'group'     # Grouping with parentheses
    
    def __init__(self, content: List['GrammarElement'], operator: str):
        self.content = content  # List of elements inside this construct
        self.operator = operator  # One of '?', '*', '+', or 'group'
    
    def __str__(self):
        content_str = ' '.join(str(elem) for elem in self.content)
        if self.operator == self.GROUP:
            return f"({content_str})"
        return f"({content_str}){self.operator}"
    
    def __repr__(self):
        return f"EBNFConstruct({self.content}, '{self.operator}')"


# Union type for all possible grammar elements
GrammarElement = Union[PDASymbol, 'EBNFConstruct']


class Production:
    """Represents a production rule in the grammar."""
    
    def __init__(self, lhs: PDASymbol, rhs: List[GrammarElement]):
        self.lhs = lhs  # Left-hand side (non-terminal)
        self.rhs = rhs  # Right-hand side (list of symbols or EBNF constructs)
    
    def __str__(self) -> str:
        rhs_str = ' '.join(str(s) for s in self.rhs) if self.rhs else 'ε'
        return f"{self.lhs} -> {rhs_str}"
    
    def __repr__(self):
        return f"Production({repr(self.lhs)}, {repr(self.rhs)})"


class ContextFreeGrammar:
    """Enhanced representation of a context-free grammar with EBNF support."""
    
    def __init__(self, 
                 non_terminals: Set[str], 
                 terminals: Set[str], 
                 productions: List[Tuple[str, List[Union[str, Tuple[List[str], str]]]]],  # Complex to support EBNF
                 start_symbol: str):
        # Convert string representations to PDASymbol objects
        self.non_terminals = {PDASymbol(nt, False) for nt in non_terminals}
        self.terminals = {PDASymbol(t, True) for t in terminals}
        
        # Storage for all symbols
        self.symbols = self.non_terminals.union(self.terminals)
        
        # The original productions with EBNF constructs
        self.ebnf_productions = []
        
        # Convert production strings to internal representation with EBNF support
        for lhs, rhs in productions:
            lhs_symbol = PDASymbol(lhs, False)
            rhs_elements = []
            
            for element in rhs:
                if isinstance(element, str):
                    # Simple terminal or non-terminal
                    is_terminal = element in terminals
                    rhs_elements.append(PDASymbol(element, is_terminal))
                elif isinstance(element, tuple):
                    # EBNF construct: (content_list, operator)
                    content, operator = element
                    # Convert content strings to symbols
                    content_elements = []
                    for item in content:
                        is_terminal = item in terminals
                        content_elements.append(PDASymbol(item, is_terminal))
                    # Create EBNF construct
                    rhs_elements.append(EBNFConstruct(content_elements, operator))
            
            self.ebnf_productions.append(Production(lhs_symbol, rhs_elements))
        
        self.start_symbol = PDASymbol(start_symbol, False)
        
        # Standard CFG productions (expanded from EBNF)
        self.productions = []
        
        # Counter for generating unique non-terminal names
        self.unique_counter = 0
    
    def expand_ebnf(self) -> 'ContextFreeGrammar':
        """
        Convert EBNF constructs to standard CFG productions.
        Returns a new grammar with only standard CFG productions.
        """
        # Create a new grammar with the same terminals and non-terminals
        expanded_grammar = ContextFreeGrammar(
            non_terminals={nt.value for nt in self.non_terminals},
            terminals={t.value for t in self.terminals},
            productions=[],  # Will be filled with expanded productions
            start_symbol=self.start_symbol.value
        )
        
        # Keep track of new non-terminals created during expansion
        new_non_terminals = set()
        
        # Process each EBNF production
        for prod in self.ebnf_productions:
            # If this production has no EBNF constructs, just add it directly
            if all(isinstance(elem, PDASymbol) for elem in prod.rhs):
                expanded_grammar.productions.append(Production(
                    prod.lhs,
                    prod.rhs
                ))
                continue
            
            # This production contains EBNF constructs, so expand them
            expanded_prods = self._expand_production(prod, new_non_terminals)
            for expanded_prod in expanded_prods:
                expanded_grammar.productions.append(expanded_prod)
        
        # Add all the new non-terminals to the grammar
        for nt in new_non_terminals:
            expanded_grammar.non_terminals.add(nt)
        
        return expanded_grammar
    
    def _expand_production(self, prod: Production, new_non_terminals: Set[PDASymbol]) -> List[Production]:
        """
        Expand a single production with EBNF constructs into multiple standard CFG productions.
        Updates new_non_terminals with any new non-terminals created.
        Returns a list of expanded productions.
        """
        # If this production has no EBNF constructs, return it directly
        if all(isinstance(elem, PDASymbol) for elem in prod.rhs):
            return [prod]
        
        result = []
        
        # Find the first EBNF construct in the production
        ebnf_index = next((i for i, elem in enumerate(prod.rhs) if isinstance(elem, EBNFConstruct)), None)
        
        if ebnf_index is not None:
            # Extract the EBNF construct
            ebnf = prod.rhs[ebnf_index]
            
            # Create a new non-terminal for this construct
            new_nt_name = f"A{self.unique_counter}"
            self.unique_counter += 1
            new_nt = PDASymbol(new_nt_name, False)
            new_non_terminals.add(new_nt)
            
            # Create a new production with the EBNF construct replaced by the new non-terminal
            new_rhs = prod.rhs[:ebnf_index] + [new_nt] + prod.rhs[ebnf_index+1:]
            new_prod = Production(prod.lhs, new_rhs)
            
            # Recursively expand the new production
            expanded_new_prods = self._expand_production(new_prod, new_non_terminals)
            result.extend(expanded_new_prods)
            
            # Create productions for the new non-terminal based on the EBNF operator
            if ebnf.operator == EBNFConstruct.OPTIONAL:  # A?
                # A? -> A | ε
                result.append(Production(new_nt, ebnf.content))  # A -> A
                result.append(Production(new_nt, []))  # A -> ε
                
            elif ebnf.operator == EBNFConstruct.STAR:  # A*
                # A* -> A A* | ε
                result.append(Production(new_nt, ebnf.content + [new_nt]))  # A -> A A
                result.append(Production(new_nt, []))  # A -> ε
                
            elif ebnf.operator == EBNFConstruct.PLUS:  # A+
                # A+ -> A A* | A
                star_nt_name = f"A{self.unique_counter}"
                self.unique_counter += 1
                star_nt = PDASymbol(star_nt_name, False)
                new_non_terminals.add(star_nt)
                
                result.append(Production(new_nt, ebnf.content + [star_nt]))  # A+ -> A A*
                result.append(Production(star_nt, ebnf.content + [star_nt]))  # A* -> A A*
                result.append(Production(star_nt, []))  # A* -> ε
                
            elif ebnf.operator == EBNFConstruct.GROUP:  # (A B C)
                # Just replace with the content
                result.append(Production(new_nt, ebnf.content))
                
        return result
    
    def is_terminal(self, symbol: PDASymbol) -> bool:
        """Check if a symbol is a terminal."""
        return symbol.is_terminal
    
    def is_non_terminal(self, symbol: PDASymbol) -> bool:
        """Check if a symbol is a non-terminal."""
        return not symbol.is_terminal
    
    def get_productions_for(self, non_terminal: PDASymbol) -> List[Production]:
        """Get all productions with the given non-terminal on the left-hand side."""
        return [p for p in self.productions if p.lhs == non_terminal]
    
    def get_rule_references(self) -> Dict[PDASymbol, Set[PDASymbol]]:
        """
        Get a dictionary of rule references.
        Keys are non-terminals, values are sets of non-terminals referenced in their productions.
        """
        references = {nt: set() for nt in self.non_terminals}
        
        for prod in self.productions:
            lhs = prod.lhs
            for element in prod.rhs:
                if isinstance(element, PDASymbol) and not element.is_terminal:
                    references[lhs].add(element)
        
        return references
    
    def detect_left_recursion(self) -> Dict[PDASymbol, bool]:
        """
        Detect direct left recursion in the grammar.
        Returns a dictionary mapping non-terminals to whether they have direct left recursion.
        """
        has_left_recursion = {nt: False for nt in self.non_terminals}
        
        for prod in self.productions:
            lhs = prod.lhs
            if prod.rhs and isinstance(prod.rhs[0], PDASymbol) and prod.rhs[0] == lhs:
                has_left_recursion[lhs] = True
        
        return has_left_recursion
    
    @classmethod
    def from_g4_grammar(cls, lexer_file: str, parser_file: str) -> 'ContextFreeGrammar':
        """
        Create a CFG from ANTLR4 grammar files.
        This would require parsing the G4 files.
        
        Placeholder for now - will be implemented later.
        """
        # This would be a complex implementation to parse G4 files
        # For now, return a placeholder with EBNF example
        return cls(
            non_terminals={'S', 'A', 'B'},
            terminals={'a', 'b', 'c'},
            productions=[
                ('S', ['A', 'B']),
                ('A', [('a', EBNFConstruct.STAR)]),  # a*
                ('A', [('a', EBNFConstruct.PLUS)]),  # a+
                ('A', [('a', EBNFConstruct.OPTIONAL)]),  # a?
                ('B', [('b', EBNFConstruct.PLUS)]),  # b+
                ('B', ['c'])
            ],
            start_symbol='S'
        )