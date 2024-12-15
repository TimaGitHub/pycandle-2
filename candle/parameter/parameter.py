def ParameterObj():
    class Parameter:
        layers = []
        calling = dict()
        def __init__(self, info):
            Parameter.layers.append(info[0])
            Parameter.calling[info[0]] = info[1:]
    return Parameter


