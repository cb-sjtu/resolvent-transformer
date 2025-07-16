#!/usr/bin/env python3

import asyncio
import json
from pathlib import Path
from typing import Any

from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    TextContent,
    Tool,
)

# run with `npx @modelcontextprotocol/inspector python logs_mcp_server.py`
# not really tested, but this file can serve as context for analyzing logs

# Initialize the MCP server
server = Server("logs-analyzer")

# Base logs directory - adjust this path as needed
LOGS_BASE_DIR = Path("logs/train/runs")


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List all available tools for analyzing logs."""
    return [
        Tool(
            name="list_runs",
            description="List all available training runs by timestamp",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="get_run_config",
            description="Get configuration details for a specific run",
            inputSchema={
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string", "description": "Run timestamp (e.g., 2025-07-12_18-11-28-389603)"}
                },
                "required": ["timestamp"],
            },
        ),
        Tool(
            name="get_run_metadata",
            description="Get system metadata and run parameters from wandb",
            inputSchema={
                "type": "object",
                "properties": {"timestamp": {"type": "string", "description": "Run timestamp"}},
                "required": ["timestamp"],
            },
        ),
        Tool(
            name="list_datasets_and_steps",
            description="List available datasets and validation steps for a run",
            inputSchema={
                "type": "object",
                "properties": {"timestamp": {"type": "string", "description": "Run timestamp"}},
                "required": ["timestamp"],
            },
        ),
        Tool(
            name="get_metrics",
            description="Get combined metrics across all GPU ranks for a specific step and dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string", "description": "Run timestamp"},
                    "step": {"type": "integer", "description": "Validation step number"},
                    "dataset": {"type": "string", "description": "Dataset name"},
                    "metric_name": {"type": "string", "description": "Metric name (e.g., 'error', 'loss')"},
                },
                "required": ["timestamp", "step", "dataset", "metric_name"],
            },
        ),
        Tool(
            name="get_sample_descriptions",
            description="Get sample descriptions for a specific step and dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string", "description": "Run timestamp"},
                    "step": {"type": "integer", "description": "Validation step number"},
                    "dataset": {"type": "string", "description": "Dataset name"},
                },
                "required": ["timestamp", "step", "dataset"],
            },
        ),
        Tool(
            name="get_batch_info",
            description="Get detailed batch information for a specific step and dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string", "description": "Run timestamp"},
                    "step": {"type": "integer", "description": "Validation step number"},
                    "dataset": {"type": "string", "description": "Dataset name"},
                },
                "required": ["timestamp", "step", "dataset"],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""

    if name == "list_runs":
        return await list_runs()
    elif name == "get_run_config":
        return await get_run_config(arguments["timestamp"])
    elif name == "get_run_metadata":
        return await get_run_metadata(arguments["timestamp"])
    elif name == "list_datasets_and_steps":
        return await list_datasets_and_steps(arguments["timestamp"])
    elif name == "get_metrics":
        return await get_metrics(
            arguments["timestamp"], arguments["step"], arguments["dataset"], arguments["metric_name"]
        )
    elif name == "get_sample_descriptions":
        return await get_sample_descriptions(arguments["timestamp"], arguments["step"], arguments["dataset"])
    elif name == "get_batch_info":
        return await get_batch_info(arguments["timestamp"], arguments["step"], arguments["dataset"])
    else:
        raise ValueError(f"Unknown tool: {name}")


async def list_runs() -> list[TextContent]:
    """List all available training runs."""
    try:
        runs = []
        if LOGS_BASE_DIR.exists():
            for run_dir in LOGS_BASE_DIR.iterdir():
                if run_dir.is_dir():
                    runs.append(run_dir.name)

        runs.sort(reverse=True)  # Most recent first

        if runs:
            result = "Available training runs:\n" + "\n".join(f"- {run}" for run in runs)
        else:
            result = "No training runs found in logs directory."

        return [TextContent(type="text", text=result)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error listing runs: {str(e)}")]


async def get_run_config(timestamp: str) -> list[TextContent]:
    """Get configuration for a specific run."""
    try:
        config_path = LOGS_BASE_DIR / timestamp / "config_tree.log"

        if not config_path.exists():
            return [TextContent(type="text", text=f"Config file not found for run {timestamp}")]

        with open(config_path) as f:
            config_content = f.read()

        return [TextContent(type="text", text=f"Configuration for run {timestamp}:\n\n{config_content}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error reading config: {str(e)}")]


async def get_run_metadata(timestamp: str) -> list[TextContent]:
    """Get wandb metadata for a specific run."""
    try:
        # Find the wandb metadata file
        wandb_dir = LOGS_BASE_DIR / timestamp / "wandb"
        metadata_files = list(wandb_dir.glob("*/files/wandb-metadata.json"))

        if not metadata_files:
            return [TextContent(type="text", text=f"Wandb metadata not found for run {timestamp}")]

        metadata_path = metadata_files[0]  # Take the first one found

        with open(metadata_path) as f:
            metadata = json.load(f)

        formatted_metadata = json.dumps(metadata, indent=2)
        return [TextContent(type="text", text=f"Metadata for run {timestamp}:\n\n{formatted_metadata}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error reading metadata: {str(e)}")]


async def list_datasets_and_steps(timestamp: str) -> list[TextContent]:
    """List available datasets and steps for a run."""
    try:
        metric_dir = LOGS_BASE_DIR / timestamp / "metric" / "valid"

        if not metric_dir.exists():
            return [TextContent(type="text", text=f"Metrics directory not found for run {timestamp}")]

        steps_and_datasets = {}

        for step_dir in metric_dir.iterdir():
            if step_dir.is_dir() and step_dir.name.startswith("step_"):
                step_num = step_dir.name
                datasets = [d.name for d in step_dir.iterdir() if d.is_dir()]
                steps_and_datasets[step_num] = datasets

        result = f"Available steps and datasets for run {timestamp}:\n\n"
        for step, datasets in sorted(steps_and_datasets.items()):
            result += f"{step}:\n"
            for dataset in datasets:
                result += f"  - {dataset}\n"
            result += "\n"

        return [TextContent(type="text", text=result)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error listing datasets and steps: {str(e)}")]


async def get_metrics(timestamp: str, step: int, dataset: str, metric_name: str) -> list[TextContent]:
    """Get combined metrics across all GPU ranks."""
    try:
        metric_dir = LOGS_BASE_DIR / timestamp / "metric" / "valid" / f"step_{step}" / dataset

        if not metric_dir.exists():
            return [TextContent(type="text", text=f"Metric directory not found: {metric_dir}")]

        # Find all rank files for this metric
        metric_files = list(metric_dir.glob(f"{metric_name}_rank*.csv"))

        if not metric_files:
            return [
                TextContent(type="text", text=f"No {metric_name} metric files found for step {step}, dataset {dataset}")
            ]

        # Combine all rank files
        all_metrics = []
        for file_path in sorted(metric_files):
            rank = file_path.stem.split("_rank")[-1]
            with open(file_path) as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if line.strip():
                        all_metrics.append({"rank": rank, "sample_idx": i, "value": float(line.strip())})

        # Create summary
        values = [m["value"] for m in all_metrics]
        summary = f"Metrics summary for {metric_name} at step {step}, dataset {dataset}:\n"
        summary += f"Total samples: {len(all_metrics)}\n"
        summary += f"Mean: {sum(values) / len(values):.6f}\n"
        summary += f"Min: {min(values):.6f}\n"
        summary += f"Max: {max(values):.6f}\n\n"

        summary += "Sample-wise values:\n"
        for metric in all_metrics[:10]:  # Show first 10 samples
            summary += f"Rank {metric['rank']}, Sample {metric['sample_idx']}: {metric['value']:.6f}\n"

        if len(all_metrics) > 10:
            summary += f"... and {len(all_metrics) - 10} more samples\n"

        return [TextContent(type="text", text=summary)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error getting metrics: {str(e)}")]


async def get_sample_descriptions(timestamp: str, step: int, dataset: str) -> list[TextContent]:
    """Get sample descriptions for a specific step and dataset."""
    try:
        metric_dir = LOGS_BASE_DIR / timestamp / "metric" / "valid" / f"step_{step}" / dataset

        if not metric_dir.exists():
            return [TextContent(type="text", text=f"Metric directory not found: {metric_dir}")]

        # Find description files
        desc_files = list(metric_dir.glob("description_rank*.txt"))

        if not desc_files:
            return [TextContent(type="text", text=f"No description files found for step {step}, dataset {dataset}")]

        result = f"Sample descriptions for step {step}, dataset {dataset}:\n\n"

        for file_path in sorted(desc_files):
            rank = file_path.stem.split("_rank")[-1]
            result += f"Rank {rank}:\n"

            with open(file_path) as f:
                content = f.read()
                result += content + "\n\n"

        return [TextContent(type="text", text=result)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error getting sample descriptions: {str(e)}")]


async def get_batch_info(timestamp: str, step: int, dataset: str) -> list[TextContent]:
    """Get batch information for a specific step and dataset."""
    try:
        batch_info_dir = LOGS_BASE_DIR / timestamp / "batch_info" / "valid" / f"step_{step}" / dataset

        if not batch_info_dir.exists():
            return [TextContent(type="text", text=f"Batch info directory not found: {batch_info_dir}")]

        # Find rank files
        rank_files = list(batch_info_dir.glob("rank_*.txt"))

        if not rank_files:
            return [TextContent(type="text", text=f"No batch info files found for step {step}, dataset {dataset}")]

        result = f"Batch information for step {step}, dataset {dataset}:\n\n"

        for file_path in sorted(rank_files):
            rank = file_path.stem.split("_")[-1]
            result += f"Rank {rank}:\n"

            with open(file_path) as f:
                content = f.read()
                result += content + "\n\n"

        return [TextContent(type="text", text=result)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error getting batch info: {str(e)}")]


async def main():
    # Run the server using stdin/stdout streams
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="logs-analyzer",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
