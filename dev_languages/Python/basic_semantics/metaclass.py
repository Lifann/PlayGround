import six

class MetaClass(type):
	def __call__(cls, *args, **kwargs):
		print('call Metaclass')
		print(cls)
		check = super(MetaClass, cls)
		#print('check: ', print(dir(check)))
	def __init__(cls, *args, **kwargs):
		print('init Metaclass')

class MyBase(object):
	def __init__(self, *args, **kwargs):
		print('init MyBase')
	
	def __new__(self, *args, **kwargs):
		print('new MyBase')

	def __call__(self, *args, **kwargs):
		print(args, kwargs)

class A(six.with_metaclass(MetaClass, MyBase)):
	def __init__(self, *args, **kwargs):
		print('init A')
	def __new__(self, *args, **kwargs):
		print('new A')
	def __call__(self, *args, **kwargs):
		print('call A')


a = A()
#print(type(a))
