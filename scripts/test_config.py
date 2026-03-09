"""
Quick test to verify configuration loads correctly.

Run: python -m scripts.test_config
"""

from rich.console import Console
from rich.table import Table

console = Console()


def main():
    try:
        from app.config import get_settings

        settings = get_settings()

        table = Table(title="🇬🇧 UK LawAssistant Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        # App settings
        table.add_row("Environment", settings.app.environment)
        table.add_row("Log Level", settings.app.log_level)
        table.add_row("App Name", settings.app.app_name)

        # Relax AI settings (mask the API key)
        key = settings.relax_ai.api_key
        masked_key = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "****"
        table.add_row("API Key", masked_key)
        table.add_row("Base URL", settings.relax_ai.base_url)
        table.add_row("Model", settings.relax_ai.model)
        table.add_row("Max Tokens", str(settings.relax_ai.max_tokens))
        table.add_row("Temperature", str(settings.relax_ai.temperature))
        table.add_row("Timeout (s)", str(settings.relax_ai.timeout))
        table.add_row("Max Retries", str(settings.relax_ai.max_retries))

        console.print(table)
        console.print("\n[bold green]✓ Configuration loaded successfully![/bold green]")

    except Exception as e:
        console.print(f"\n[bold red]✗ Configuration error: {e}[/bold red]")
        raise


if __name__ == "__main__":
    main()
