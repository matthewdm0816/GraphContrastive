from typing import MethodType, List, Optional

class Config:
    def __init__(self):
        pass
    
    @staticmethod
    def parse_to_dict(self, args):
        args_dict = {}
        for arg in dir(args):
            if not arg.startswith('_') and not isinstance(getattr(args, arg), MethodType):
                if getattr(args, arg) is not None:
                    args_dict[arg] = getattr(args, arg)

        return args_dict
    
    def add_args(self, args_dict):
        for arg in args_dict:
            setattr(self, arg, args_dict[arg])

    def __str__(self):
        res = []
        for attr in dir(self):
            if not attr.startswith('__') and not isinstance(getattr(self, attr), MethodType):
                res.append(('{ %-17s }->' % attr) + getattr(self, attr))

        return '\n'.join(res)
    
    def parse_from_yml(self, filename):
        import yaml
        with open(filename, 'r') as f:
            args = yaml.safe_load(f)
        self.add_args(args)