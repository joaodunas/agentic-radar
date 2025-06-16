import ast
import re
from typing import Union

from pydantic import ValidationError

from ...ast_utils import (
    get_keyword_arg_value,
    get_nth_arg_value,
    get_simple_identifier_name,
    get_string_keyword_arg,
    is_function_call,
    is_simple_identifier,
)
from ...utils import walk_python_files
from ..exceptions import InvalidPydanticAIAgentConstructorError
from ..models import PydanticAIAgent, PydanticAITool, SystemPrompt, OutputType, DependencyInfo


class PydanticAIAgentsVisitor(ast.NodeVisitor):
    AGENT_CLASS_NAME = "Agent"
    
    def __init__(self):
        super().__init__()
        self.agent_assignments: dict[str, PydanticAIAgent] = {}
        self.tools_by_agent: dict[str, list[PydanticAITool]] = {}
        self.system_prompts_by_agent: dict[str, list[SystemPrompt]] = {}

    def visit_Assign(self, node):
        if is_function_call(node.value, self.AGENT_CLASS_NAME):
            self._visit_agent_assignment(node)

    def visit_FunctionDef(self, node):
        self._check_for_decorators(node)

    def visit_AsyncFunctionDef(self, node):
        self._check_for_decorators(node)

    def _visit_agent_assignment(self, node: ast.Assign) -> None:
        assert isinstance(node.value, ast.Call)

        try:
            agent = self._extract_agent(node.value)
        except InvalidPydanticAIAgentConstructorError as e:
            print(f"Invalid Agent constructor: {ast.dump(node.value)}. Error: {e}")
            return

        for target in node.targets:
            if not is_simple_identifier(target):
                continue

            target_name = get_simple_identifier_name(target)
            self.agent_assignments[target_name] = agent

    def _check_for_decorators(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]):
        for decorator in node.decorator_list:
            if self._is_tool_decorator(decorator):
                self._extract_tool_from_decorator(node, decorator)
            elif self._is_system_prompt_decorator(decorator):
                self._extract_system_prompt_from_decorator(node, decorator)

    def _is_tool_decorator(self, decorator: ast.AST) -> bool:
        if isinstance(decorator, ast.Attribute):
            return decorator.attr == "tool"
        elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
            return decorator.func.attr == "tool"
        return False

    def _is_system_prompt_decorator(self, decorator: ast.AST) -> bool:
        if isinstance(decorator, ast.Attribute):
            return decorator.attr == "system_prompt"
        elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
            return decorator.func.attr == "system_prompt"
        return False

    def _extract_agent(self, agent_node: ast.Call) -> PydanticAIAgent:
        try:
            model = self._extract_model(agent_node)
            model_provider, model_name = self._parse_model_string(model)
            deps_type = self._extract_deps_type(agent_node)
            output_type = self._extract_output_type(agent_node)
            static_system_prompt = self._extract_static_system_prompt(agent_node)
            
            # Create a temporary agent name based on assignment (will be updated later)
            agent_name = "unknown_agent"
            
            # Create initial system prompts list with static prompt if present
            system_prompts = []
            if static_system_prompt:
                system_prompts.append(static_system_prompt)
            
            return PydanticAIAgent(
                name=agent_name,
                model=model,
                model_provider=model_provider,
                model_name=model_name,
                deps_type=deps_type,
                output_type=output_type,
                system_prompts=system_prompts,
                uses_dependency_injection=deps_type is not None,
                uses_structured_output=output_type is not None
            )
        except (ValueError, ValidationError) as e:
            raise InvalidPydanticAIAgentConstructorError from e

    def _extract_model(self, agent_node: ast.Call) -> str:
        # Model can be first positional argument or 'model' keyword
        model_node = get_nth_arg_value(agent_node, 0)
        if not model_node:
            model_node = get_keyword_arg_value(agent_node, "model")
        
        if not model_node:
            return "gpt-4o"  # Default
            
        if isinstance(model_node, ast.Constant) and isinstance(model_node.value, str):
            return model_node.value
        elif isinstance(model_node, ast.Str):
            return model_node.s
        else:
            return "unknown"

    def _parse_model_string(self, model: str) -> tuple[str, str]:
        if ":" in model:
            provider, model_name = model.split(":", 1)
            return provider, model_name
        return "unknown", model

    def _extract_deps_type(self, agent_node: ast.Call) -> str:
        deps_type_node = get_keyword_arg_value(agent_node, "deps_type")
        if not deps_type_node:
            return None
            
        if isinstance(deps_type_node, ast.Name):
            return deps_type_node.id
        elif isinstance(deps_type_node, ast.Attribute):
            return f"{deps_type_node.value.id}.{deps_type_node.attr}" if isinstance(deps_type_node.value, ast.Name) else deps_type_node.attr
        else:
            return str(deps_type_node.__class__.__name__)

    def _extract_output_type(self, agent_node: ast.Call) -> OutputType:
        output_type_node = get_keyword_arg_value(agent_node, "output_type")
        if not output_type_node:
            return None
            
        type_name = self._ast_to_string(output_type_node)
        return OutputType(type_name=type_name, is_structured=True)
    
    def _ast_to_string(self, node: ast.AST) -> str:
        """Convert an AST node to a string representation of the type."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value_str = self._ast_to_string(node.value) if hasattr(node, 'value') else "unknown"
            return f"{value_str}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            # Handle subscript types like list[Type], dict[str, int], etc.
            value_str = self._ast_to_string(node.value)
            slice_str = self._ast_to_string(node.slice)
            return f"{value_str}[{slice_str}]"
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            # Handle union types like Type1 | Type2
            left_str = self._ast_to_string(node.left)
            right_str = self._ast_to_string(node.right)
            return f"{left_str} | {right_str}"
        elif isinstance(node, ast.Tuple):
            # Handle tuple types
            elements = [self._ast_to_string(elt) for elt in node.elts]
            return f"({', '.join(elements)})"
        elif isinstance(node, ast.List):
            # Handle list literals in type annotations
            elements = [self._ast_to_string(elt) for elt in node.elts]
            return f"[{', '.join(elements)}]"
        elif isinstance(node, ast.Constant):
            # Handle constant values (strings, numbers, etc.)
            return repr(node.value)
        elif isinstance(node, ast.Str):
            # Handle string literals (Python < 3.8)
            return repr(node.s)
        else:
            # Fallback: try to get the class name
            return getattr(node, 'id', node.__class__.__name__)

    def _extract_static_system_prompt(self, agent_node: ast.Call) -> SystemPrompt:
        system_prompt_node = get_keyword_arg_value(agent_node, "system_prompt")
        if not system_prompt_node:
            return None
            
        prompt_content = None
        if isinstance(system_prompt_node, ast.Constant) and isinstance(system_prompt_node.value, str):
            prompt_content = system_prompt_node.value
        elif isinstance(system_prompt_node, ast.Str):
            prompt_content = system_prompt_node.s
        elif isinstance(system_prompt_node, (ast.JoinedStr, ast.FormattedValue)):
            # Handle f-strings or other complex string expressions
            prompt_content = "Complex string expression"
        elif isinstance(system_prompt_node, ast.BinOp) and isinstance(system_prompt_node.op, ast.Add):
            # Handle string concatenation
            prompt_content = "Concatenated string expression"
        else:
            # Try to extract from other types if possible
            prompt_content = "Static system prompt"
            
        if prompt_content:
            return SystemPrompt(
                type='static',
                content=prompt_content,
                uses_dependencies=False,
                dependency_type=None
            )
        
        return None

    def _extract_tool_from_decorator(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], decorator: ast.AST):
        agent_name = self._get_agent_name_from_decorator(decorator)
        if not agent_name:
            return
            
        tool = PydanticAITool(
            name=node.name,
            function_name=node.name,
            description=ast.get_docstring(node) or "",
            agent_name=agent_name,
            uses_run_context=self._uses_run_context(node),
            dependency_type=self._extract_run_context_type(node)
        )
        tool._function_def = node
        
        if agent_name not in self.tools_by_agent:
            self.tools_by_agent[agent_name] = []
        self.tools_by_agent[agent_name].append(tool)

    def _extract_system_prompt_from_decorator(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], decorator: ast.AST):
        agent_name = self._get_agent_name_from_decorator(decorator)
        if not agent_name:
            return
            
        prompt = SystemPrompt(
            type='dynamic',
            function_name=node.name,
            uses_dependencies=self._uses_run_context(node),
            dependency_type=self._extract_run_context_type(node)
        )
        prompt._function_def = node
        
        if agent_name not in self.system_prompts_by_agent:
            self.system_prompts_by_agent[agent_name] = []
        self.system_prompts_by_agent[agent_name].append(prompt)

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


def collect_pydantic_agent_assignments(root_dir: str) -> dict[str, PydanticAIAgent]:
    all_agent_assignments: dict[str, PydanticAIAgent] = {}
    
    for file in walk_python_files(root_dir):
        with open(file, "r") as f:
            try:
                tree = ast.parse(f.read())
            except Exception as e:
                print(f"Cannot parse Python module: {file}. Error: {e}")
                continue
                
            visitor = PydanticAIAgentsVisitor()
            visitor.visit(tree)
            
            # Merge agents and associate tools/system prompts
            for agent_name, agent in visitor.agent_assignments.items():
                agent.name = agent_name
                agent.tools = visitor.tools_by_agent.get(agent_name, [])
                
                # Merge static system prompts (from constructor) with dynamic ones (from decorators)
                dynamic_system_prompts = visitor.system_prompts_by_agent.get(agent_name, [])
                agent.system_prompts.extend(dynamic_system_prompts)
                
                all_agent_assignments[agent_name] = agent
    
    return all_agent_assignments 