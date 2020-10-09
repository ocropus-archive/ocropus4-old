import click

@click.command()
@click.option("--arg", default=17)
def command(arg=19):
    print(arg, type(arg))

if __name__=='__main__':
    command()
