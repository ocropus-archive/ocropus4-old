from invoke import task


@task
def test1(c, arg1, opt="q"):
    """test1 is a test command.

        :param arg1: arg1
        :param opt: some option
    """
    print(c, arg1, opt)

@task
def test2(c, something, somethingelse="abc"):
    print(c, something, somethingelse)
