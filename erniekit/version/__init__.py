
import os

from . import git

commit = "unknown"

erniekit_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if commit.endswith("unknown") and git.is_git_repo(erniekit_dir) and git.have_git():
    commit = git.git_revision(erniekit_dir).decode("utf-8")
    if git.is_dirty(erniekit_dir):
        commit += ".dirty"
del erniekit_dir


__all__ = ["show"]


def show():
    """Get the corresponding commit id of erniekit.

    Returns:
        The commit-id of erniekit will be output.

        full_version: version of erniekit


    Examples:
        .. code-block:: python

            import erniekit

            erniekit.version.show()
            # commit: 1ef5b94a18773bb0b1bba1651526e5f5fc5b16fa

    """
    print("commit:", commit)