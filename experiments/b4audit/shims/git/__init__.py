"""Shim for tpp.utils.run: records a fixed commit identifier instead of querying the repository."""
class _Commit:
    hexsha = 'ntpp-tmlr2023@54c15fd'
class _Head:
    object = _Commit()
class Repo:
    def __init__(self, *a, **k):
        self.head = _Head()

    def is_dirty(self):
        return False
