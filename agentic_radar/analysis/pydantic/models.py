import ast
from enum import Enum
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, PrivateAttr


class SystemPrompt(BaseModel):
    type: Literal['static', 'dynamic']
    content: Optional[str] = None
    function_name: Optional[str] = None
    _function_def: Optional[Union[ast.FunctionDef, ast.AsyncFunctionDef]] = PrivateAttr(default=None)
    uses_dependencies: bool = False
    dependency_type: Optional[str] = None


class PydanticAITool(BaseModel):
    name: str
    function_name: str
    description: Optional[str] = None
    _function_def: Optional[Union[ast.FunctionDef, ast.AsyncFunctionDef]] = PrivateAttr(default=None)
    uses_run_context: bool = False
    dependency_type: Optional[str] = None
    agent_name: Optional[str] = None


class OutputType(BaseModel):
    type_name: str
    is_pydantic_model: bool = False
    _model_definition: Optional[ast.ClassDef] = PrivateAttr(default=None)
    is_structured: bool = True


class DependencyInfo(BaseModel):
    type_name: str
    is_dataclass: bool = False
    is_pydantic_model: bool = False
    _class_definition: Optional[ast.ClassDef] = PrivateAttr(default=None)


class PydanticAIAgent(BaseModel):
    name: str
    model: Optional[str] = None
    model_provider: Optional[str] = None
    model_name: Optional[str] = None
    deps_type: Optional[str] = None
    dependency_info: Optional[DependencyInfo] = None
    output_type: Optional[OutputType] = None
    system_prompts: list[SystemPrompt] = Field(default_factory=list)
    tools: list[PydanticAITool] = Field(default_factory=list)
    uses_structured_output: bool = False
    uses_dependency_injection: bool = False 