from importlib import import_module


class DummyImport(object):

    def __init__(self, message='haha'):
        self.message = message

    def __getattribute__(self, attr):
        if attr.startswith('__') or attr == 'message':
            return object.__getattribute__(self, attr)
        else:
            raise ImportError(object.__getattribute__(self, 'message'))

    def __call__(self, *args, **kwargs):
        raise ImportError(object.__getattribute__(self, 'message'))


def optional_import(modulename, package=None):
    try:
        if package is not None:
            return getattr(import_module(package), modulename)
        else:
            return import_module(modulename)
    except:
        if package is not None:
            fullname = package + '.' + modulename
        else:
            fullname = modulename
        return DummyImport("Module '{0}' needed for the requested functionality."
                           .format(fullname))


class AttrDict(dict):
    """
    Dictionary that allows accessing keys as attributes
    assert 11 == AttrDict(a=11, b=22).a
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
