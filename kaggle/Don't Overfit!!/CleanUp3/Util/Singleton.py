class Singleton(type) :
    _instance = None
    def __call__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instance