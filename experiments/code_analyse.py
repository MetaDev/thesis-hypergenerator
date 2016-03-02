# -*- coding: utf-8 -*-

from timeit import Timer
# first argument is the code to be run, the second "setup" argument is only run once,
# and it not included in the execution time.
t = Timer("""search_space.test()""", setup="""import search_space""")
