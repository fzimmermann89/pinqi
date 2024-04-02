def get_code_jupyter(obj):
    # https://stackoverflow.com/questions/51566497/getting-the-source-of-an-object-defined-in-a-jupyter-notebook/65806216#65806216
    obj = obj.__class__
    import inspect, sys
    from IPython.core.magics.code import extract_symbols

    def new_getfile(object, _old_getfile=inspect.getfile):
        if not inspect.isclass(object):
            return _old_getfile(object)
        if hasattr(object, "__module__"):
            object_ = sys.modules.get(object.__module__)
            if hasattr(object_, "__file__"):
                return object_.__file__
        for name, member in inspect.getmembers(object):
            if inspect.isfunction(member) and object.__qualname__ + "." + member.__name__ == member.__qualname__:
                return inspect.getfile(member)
        else:
            raise TypeError("Source for {!r} not found".format(object))

    inspect.getfile = new_getfile

    cell_code = "".join(inspect.linecache.getlines(new_getfile(obj)))
    class_code = extract_symbols(cell_code, obj.__name__)[0][0]

    return class_code
