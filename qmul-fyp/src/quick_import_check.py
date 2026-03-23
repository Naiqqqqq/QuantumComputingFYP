import os, sys
print('script dir:', os.path.dirname(__file__))
print('cwd:', os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))
print('first path:', sys.path[0])
print('contains arith?:', os.path.isdir(os.path.join(os.path.dirname(__file__), 'arith')))
try:
    import arith
    print('arith imported, file:', getattr(arith, '__file__', None))
    print('arith all:', getattr(arith, '__all__', None))
except Exception as e:
    print('import arith failed:', repr(e))
