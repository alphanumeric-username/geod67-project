class Test:
    def __getitem__(self, key):
        print(key, type(key))
        if type(key) == tuple:
            for ki in key:
                if type(ki) == slice:
                    print(ki.start)
                    print(ki.stop)
                    print(ki.step)


t = Test()

t[1]

t[:, 1]

t[2, 1]

t[1:10, 1]