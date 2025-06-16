from pathlib import Path

from agentic_radar.analysis.analyze import Analyzer
from agentic_radar.analysis.pydantic.graph import create_pydantic_graph_definition
from agentic_radar.analysis.pydantic.parsing import (
    collect_pydantic_agent_assignments,
    collect_pydantic_tool_assignments,
    collect_system_prompts,
    analyze_dependencies,
)
from agentic_radar.analysis.openai_agents.tool_categorizer.categorizer import (
    load_tool_categories,
)
from agentic_radar.graph import GraphDefinition


class PydanticAIAnalyzer(Analyzer):
    def __init__(self):
        super().__init__()

    def analyze(self, root_directory: str) -> GraphDefinition:
        print(f"Starting PydanticAI analysis of {root_directory}")
        
        # Step 1: Collect PydanticAI agent assignments (includes tools and system prompts)
        print("Collecting PydanticAI agent assignments...")
        agent_assignments = collect_pydantic_agent_assignments(root_directory)
        print(f"Found {len(agent_assignments)} PydanticAI agents")
        
        # Step 2: Analyze dependencies
        print("Analyzing dependencies...")
        dependency_info = analyze_dependencies(root_directory, agent_assignments)
        print(f"Found {len(dependency_info)} dependency types")
        
        # Step 3: Load tool categories (reuse from OpenAI analyzer)
        print("Loading tool categories...")
        tool_categories = load_tool_categories()
        
        # Step 4: Generate graph definition
        print("Generating graph definition...")
        graph_definition = create_pydantic_graph_definition(
            graph_name=Path(root_directory).name,
            agent_assignments=agent_assignments,
            tool_categories=tool_categories,
        )
        
        print(f"Analysis complete. Generated graph with {len(graph_definition.nodes)} nodes and {len(graph_definition.edges)} edges")
        return graph_definition 