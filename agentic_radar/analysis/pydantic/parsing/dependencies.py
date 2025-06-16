import ast
from typing import Union

from ...utils import walk_python_files
from ..models import DependencyInfo, PydanticAIAgent


class DependencyAnalyzer:
    def __init__(self):
        self.dependency_definitions: dict[str, DependencyInfo] = {}

    def analyze_dependencies(self, root_dir: str, agents: dict[str, PydanticAIAgent]) -> dict[str, DependencyInfo]:
        # First pass: collect all class definitions that could be dependencies
        self._collect_dependency_definitions(root_dir)
        
        # Second pass: associate dependency info with agents
        for agent in agents.values():
            if agent.deps_type and agent.deps_type in self.dependency_definitions:
                agent.dependency_info = self.dependency_definitions[agent.deps_type]
        
        return self.dependency_definitions

    def _collect_dependency_definitions(self, root_dir: str):
        for file in walk_python_files(root_dir):
            with open(file, "r") as f:
                try:
                    tree = ast.parse(f.read())
                except Exception as e:
                    print(f"Cannot parse Python module: {file}. Error: {e}")
                    continue
                    
                visitor = DependencyDefinitionsVisitor()
                visitor.visit(tree)
                self.dependency_definitions.update(visitor.dependency_definitions)


class DependencyDefinitionsVisitor(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.dependency_definitions: dict[str, DependencyInfo] = {}

    def visit_ClassDef(self, node: ast.ClassDef):
        if self._is_potential_dependency_class(node):
            dependency_info = self._extract_dependency_info(node)
            if dependency_info:
                self.dependency_definitions[node.name] = dependency_info
        self.generic_visit(node)

    def _is_potential_dependency_class(self, node: ast.ClassDef) -> bool:
        # Check if it's a dataclass
        if self._has_dataclass_decorator(node):
            return True
            
        # Check if it inherits from BaseModel (Pydantic)
        if self._inherits_from_base_model(node):
            return True
            
        # Could be a regular class used as dependency
        return True

    def _extract_dependency_info(self, node: ast.ClassDef) -> DependencyInfo:
        is_dataclass = self._has_dataclass_decorator(node)
        is_pydantic_model = self._inherits_from_base_model(node)
        
        info = DependencyInfo(
            type_name=node.name,
            is_dataclass=is_dataclass,
            is_pydantic_model=is_pydantic_model
        )
        info._class_definition = node
        return info

    def _has_dataclass_decorator(self, node: ast.ClassDef) -> bool:
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == "dataclass":
                return True
            elif isinstance(decorator, ast.Attribute) and decorator.attr == "dataclass":
                return True
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name) and decorator.func.id == "dataclass":
                    return True
                elif isinstance(decorator.func, ast.Attribute) and decorator.func.attr == "dataclass":
                    return True
        return False

    def _inherits_from_base_model(self, node: ast.ClassDef) -> bool:
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == "BaseModel":
                return True
            elif isinstance(base, ast.Attribute) and base.attr == "BaseModel":
                return True
        return False


def analyze_dependencies(root_dir: str, agents: dict[str, PydanticAIAgent]) -> dict[str, DependencyInfo]:
    analyzer = DependencyAnalyzer()
    return analyzer.analyze_dependencies(root_dir, agents) 