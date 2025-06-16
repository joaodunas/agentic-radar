import ast
from typing import Union

from ...utils import walk_python_files
from ..models import SystemPrompt


class SystemPromptsVisitor(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.system_prompts: dict[str, list[SystemPrompt]] = {}

    def visit_FunctionDef(self, node):
        self._check_for_system_prompt_decorator(node)

    def visit_AsyncFunctionDef(self, node):
        self._check_for_system_prompt_decorator(node)

    def _check_for_system_prompt_decorator(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]):
        for decorator in node.decorator_list:
            if self._is_system_prompt_decorator(decorator):
                prompt = self._extract_system_prompt_from_decorator(node, decorator)
                if prompt:
                    agent_name = self._get_agent_name_from_decorator(decorator)
                    if agent_name:
                        if agent_name not in self.system_prompts:
                            self.system_prompts[agent_name] = []
                        self.system_prompts[agent_name].append(prompt)

    def _is_system_prompt_decorator(self, decorator: ast.AST) -> bool:
        if isinstance(decorator, ast.Attribute):
            return decorator.attr == "system_prompt"
        elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
            return decorator.func.attr == "system_prompt"
        return False

    def _extract_system_prompt_from_decorator(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], decorator: ast.AST) -> SystemPrompt:
        prompt = SystemPrompt(
            type='dynamic',
            function_name=node.name,
            uses_dependencies=self._uses_run_context(node),
            dependency_type=self._extract_run_context_type(node)
        )
        prompt._function_def = node
        return prompt

    def _get_agent_name_from_decorator(self, decorator: ast.AST) -> str:
        if isinstance(decorator, ast.Attribute) and isinstance(decorator.value, ast.Name):
            return decorator.value.id
        elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute) and isinstance(decorator.func.value, ast.Name):
            return decorator.func.value.id
        return None

    def _uses_run_context(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        for arg in node.args.args:
            if arg.annotation and self._is_run_context_annotation(arg.annotation):
                return True
        return False

    def _is_run_context_annotation(self, annotation: ast.AST) -> bool:
        if isinstance(annotation, ast.Name):
            return annotation.id == "RunContext"
        elif isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name):
            return annotation.value.id == "RunContext"
        elif isinstance(annotation, ast.Attribute):
            return annotation.attr == "RunContext"
        return False

    def _extract_run_context_type(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> str:
        for arg in node.args.args:
            if arg.annotation and self._is_run_context_annotation(arg.annotation):
                if isinstance(arg.annotation, ast.Subscript):
                    type_arg = arg.annotation.slice
                    if isinstance(type_arg, ast.Name):
                        return type_arg.id
                    elif isinstance(type_arg, ast.Attribute):
                        return f"{type_arg.value.id}.{type_arg.attr}" if isinstance(type_arg.value, ast.Name) else type_arg.attr
        return None


def collect_system_prompts(root_dir: str) -> dict[str, list[SystemPrompt]]:
    all_system_prompts: dict[str, list[SystemPrompt]] = {}
    
    for file in walk_python_files(root_dir):
        with open(file, "r") as f:
            try:
                tree = ast.parse(f.read())
            except Exception as e:
                print(f"Cannot parse Python module: {file}. Error: {e}")
                continue
                
            visitor = SystemPromptsVisitor()
            visitor.visit(tree)
            
            # Merge system prompts by agent
            for agent_name, prompts in visitor.system_prompts.items():
                if agent_name not in all_system_prompts:
                    all_system_prompts[agent_name] = []
                all_system_prompts[agent_name].extend(prompts)

    return all_system_prompts 