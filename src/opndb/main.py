import click

# This creates the cli function that pyproject.toml is looking for
@click.group()
def cli():
    """OPNDB command line interface"""
    pass

# This makes "start" a subcommand of cli (ex: running "opndb start" in command line executes start() command)
@cli.command()
def start():
    """Start the OPNDB workflow"""
    click.echo("hello world")

if __name__ == "__main__":
    cli()
