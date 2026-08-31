"""
Monitor a Gemini fine-tuning job's progress.

Usage:
    python3 scripts/monitor.py <job_name>
    python3 scripts/monitor.py <job_name> --watch   # poll every 30s until done

Requires GEMINI_API_KEY environment variable to be set.
"""

import os
import sys
import time
from google import genai
from rich.console import Console

console = Console()


def show_status(client, job_name):
    job = client.tunings.get(name=job_name)

    console.print(f"\n[bold]Job:[/bold]    {job.name}")
    console.print(f"[bold]Status:[/bold] {job.state}")

    if job.tuned_model and job.tuned_model.model:
        console.print(f"[bold green]Fine-tuned model:[/bold green] {job.tuned_model.model}")

    if job.tuned_model and job.tuned_model.display_name:
        console.print(f"[bold]Display name:[/bold] {job.tuned_model.display_name}")

    # Show snapshots (training progress) if available
    if hasattr(job, "tuning_task") and job.tuning_task:
        task = job.tuning_task
        if hasattr(task, "snapshots") and task.snapshots:
            console.print(f"\n[bold]Training snapshots:[/bold]")
            for snap in task.snapshots[-10:]:
                epoch = getattr(snap, "epoch", "?")
                step = getattr(snap, "step", "?")
                loss = getattr(snap, "mean_loss", None)
                loss_str = f"{loss:.4f}" if loss is not None else "?"
                console.print(f"  Epoch {epoch}, Step {step}: loss = {loss_str}")

    return job


def main():
    if len(sys.argv) < 2:
        # If no job name given, list all tuning jobs
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("ERROR: Set GEMINI_API_KEY first.")
            sys.exit(1)
        client = genai.Client(api_key=api_key)

        console.print("[bold]Recent tuning jobs:[/bold]")
        for job in client.tunings.list():
            console.print(f"  {job.name}  —  {job.state}")
        print("\nUsage: python3 scripts/monitor.py <job_name> [--watch]")
        sys.exit(0)

    job_name = sys.argv[1]
    watch = "--watch" in sys.argv

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY first.")
        sys.exit(1)
    client = genai.Client(api_key=api_key)

    if watch:
        console.print("[bold]Watching job progress (Ctrl+C to stop)...[/bold]")
        while True:
            job = show_status(client, job_name)
            state = str(job.state)
            if "SUCCEEDED" in state or "COMPLETED" in state:
                model_name = job.tuned_model.model if job.tuned_model else "unknown"
                console.print(f"\n[bold green]Done! Your fine-tuned model: {model_name}[/bold green]")
                console.print("Use this model ID in evaluate.py to test it.")
                break
            elif "FAILED" in state or "CANCELLED" in state:
                console.print(f"\n[bold red]Job ended with status: {state}[/bold red]")
                break
            time.sleep(30)
    else:
        show_status(client, job_name)


if __name__ == "__main__":
    main()
