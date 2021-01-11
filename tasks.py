from invoke import task
from rope.base.project import Project
from rope.refactor.occurrences import Finder
from rope.refactor import move

PROJECT = Project('.')

@task
def cmove(ctxt, class_name, source, target, do=False):
    """
    Move class: --class-name <> --source <module> --target <module> [--do False]
    """
    finder = Finder(PROJECT, class_name)
    source_resource = PROJECT.get_resource(source)
    target_resource = PROJECT.get_resource(target)
    class_occurrence = next(
        occ for occ in finder.find_occurrences(resource=source_resource)
        if occ.is_defined())
    mover = move.create_move(PROJECT, source_resource, class_occurrence.offset)
    changes = mover.get_changes(target_resource)
    PROJECT.do(changes)
