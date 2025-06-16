import json
from pathlib import Path

from agentic_radar.analysis.pydantic.models import PydanticAIAgent, PydanticAITool
from agentic_radar.graph import (
    Agent as ReportAgent,
    EdgeDefinition,
    GraphDefinition,
    NodeDefinition,
    NodeType,
    ToolType,
)


def create_pydantic_graph_definition(
    graph_name: str,
    agent_assignments: dict[str, PydanticAIAgent],
    tool_categories: dict[str, ToolType],
) -> GraphDefinition:
    nodes = []
    edges = []
    tools = []

    # Create agent nodes
    for agent in agent_assignments.values():
        nodes.append(_get_pydantic_agent_node(agent))

    # Create tool and system prompt nodes, and their edges
    for agent in agent_assignments.values():
        # Handle tools
        for tool in agent.tools:
            category = tool_categories.get(tool.name, ToolType.DEFAULT)
            tool_node = _get_pydantic_tool_node(tool, category)
            nodes.append(tool_node)
            tools.append(tool_node.model_copy(deep=True))
            
            # Agent -> Tool edge
            edges.append(
                EdgeDefinition(
                    start=agent.name, 
                    end=tool.name, 
                    condition="tool_call"
                )
            )
            # Tool -> Agent edge
            edges.append(EdgeDefinition(start=tool.name, end=agent.name))

        # Note: System prompt nodes removed from graph visualization

        # Handle dependency injection relationships
        if agent.uses_dependency_injection and agent.dependency_info:
            dep_node = _get_dependency_node(agent.dependency_info)
            nodes.append(dep_node)
            
            # Dependency -> Agent edge
            edges.append(
                EdgeDefinition(
                    start=agent.dependency_info.type_name,
                    end=agent.name,
                    condition="dependency_injection"
                )
            )

        # Note: Output types are now shown in the agents table instead of as graph nodes

    # Add start and end nodes (PydanticAI doesn't have handoffs, so simpler flow)
    nodes, edges = _add_start_end_nodes_pydantic(nodes=nodes, edges=edges)

    # Create report agents
    report_agents = [
        ReportAgent(
            name=agent.name,
            llm=agent.model or "gpt-4o",
            system_prompt=_get_combined_system_prompt(agent),
            output_type=agent.output_type.type_name if agent.output_type else None,
            is_guardrail=False,  # PydanticAI doesn't have explicit guardrails
            vulnerabilities=[]  # Vulnerability analysis disabled for PydanticAI
        )
        for agent in agent_assignments.values()
    ]

    return GraphDefinition(
        name=graph_name, 
        nodes=nodes, 
        edges=edges, 
        tools=tools, 
        agents=report_agents
    )


def _get_pydantic_agent_node(agent: PydanticAIAgent) -> NodeDefinition:
    description = {
        "model": agent.model,
        "model_provider": agent.model_provider,
        "model_name": agent.model_name,
        "deps_type": agent.deps_type,
        "uses_dependency_injection": agent.uses_dependency_injection,
        "uses_structured_output": agent.uses_structured_output,
        "output_type": agent.output_type.type_name if agent.output_type else None,
    }
    
    # Use AGENT_DYNAMIC node type if the agent has dynamic system prompts
    has_dynamic_prompts = any(prompt.type == 'dynamic' for prompt in agent.system_prompts)
    node_type = NodeType.AGENT_DYNAMIC if has_dynamic_prompts else NodeType.AGENT
    
    return NodeDefinition(
        type=node_type, 
        name=agent.name, 
        label=agent.name,
        description=json.dumps(description)
    )


def _get_pydantic_tool_node(tool: PydanticAITool, category: ToolType) -> NodeDefinition:
    description = {
        "function_name": tool.function_name,
        "uses_run_context": tool.uses_run_context,
        "dependency_type": tool.dependency_type,
        "agent_name": tool.agent_name,
    }
    
    return NodeDefinition(
        type=NodeType.CUSTOM_TOOL,  # PydanticAI tools are always custom
        name=tool.name,
        description=tool.description,
        label=tool.name,
        category=category,
        metadata=description
    )





def _get_dependency_node(dependency_info) -> NodeDefinition:
    description = {
        "is_dataclass": dependency_info.is_dataclass,
        "is_pydantic_model": dependency_info.is_pydantic_model,
    }
    
    return NodeDefinition(
        type=NodeType.BASIC,
        name=dependency_info.type_name,
        label=f"Dependency: {dependency_info.type_name}",
        description=json.dumps(description)
    )





def _get_combined_system_prompt(agent: PydanticAIAgent) -> str:
    static_prompts = [p.content for p in agent.system_prompts if p.type == 'static' and p.content]
    dynamic_prompts = [f"Dynamic prompt: {p.function_name}" for p in agent.system_prompts if p.type == 'dynamic']
    
    all_prompts = static_prompts + dynamic_prompts
    return " | ".join(all_prompts) if all_prompts else "No explicit system prompt"


def _add_start_end_nodes_pydantic(nodes: list[NodeDefinition], edges: list[EdgeDefinition]):
    """
    Add start and end nodes to the PydanticAI graph.
    Since PydanticAI doesn't have handoffs, we look at dependency injection patterns.
    """
    # Create start and end nodes
    start_node = NodeDefinition(type=NodeType.BASIC, name="start", label="Start")
    end_node = NodeDefinition(type=NodeType.BASIC, name="end", label="End")

    # Find agent node names (both regular and dynamic agents)
    agent_nodes = {node.name for node in nodes if node.node_type in (NodeType.AGENT, NodeType.AGENT_DYNAMIC)}

    # In PydanticAI, agents typically don't chain to each other, so every agent gets start/end
    for agent_name in agent_nodes:
        # Start -> Agent
        edges.append(EdgeDefinition(start="start", end=agent_name))
        # Agent -> End
        edges.append(EdgeDefinition(start=agent_name, end="end"))

    # Add start and end nodes to the node list
    nodes.extend([start_node, end_node])

    return nodes, edges 